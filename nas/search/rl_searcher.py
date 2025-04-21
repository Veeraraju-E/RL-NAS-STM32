import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from search.base_searcher import BaseSearcher
from search.networks import PolicyNetwork, ValueNetwork
from search.experience_buffer import ExperienceBuffer

class RLSearcher(BaseSearcher):
    def __init__(self, search_space, hardware_constraints, evaluator, device):
        super().__init__(search_space, hardware_constraints)
        self.evaluator = evaluator
        self.device = device
        self.experience_buffer = ExperienceBuffer()
        
        # space to action dimensions
        self.action_dims = self._create_action_dims()
        
        # State dimension
        self.state_dim = len(self.search_space) + len(self.hw_constraints)

        # networks
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dims).to(device)
        self.value_net = ValueNetwork(self.state_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        # PPO hyperparams
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.ppo_epochs = 10
        self.batch_size = 32
        
        self.search_history = []
        self.best_architectures = []        
        self.best_architecture = {}
        for param, values in self.search_space.items():
            if isinstance(values, tuple):
                self.best_architecture[param] = (values[0] + values[1]) / 2
            else:
                self.best_architecture[param] = values[len(values)//2]
    
    def search(self, num_iterations):
        """Execute RL-based architecture search"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("search_results") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Starting RL-based architecture search...")
        
        for iteration in range(num_iterations):
            # evaluate generated architecture
            state = self._get_current_state()
            architecture, log_probs = self._sample_architecture(state)
            score = self.evaluator.evaluate_architecture(architecture)
            
            # experience
            self.experience_buffer.add(
                state=state,
                actions=self._architecture_to_actions(architecture),
                reward=score,
                log_probs=log_probs
            )
            
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
                self._update_best_architectures(architecture, score)
            
            logging.info(f"\nIteration {iteration + 1}/{num_iterations}")
            logging.info(f"Current architecture: {architecture}")
            logging.info(f"Score: {score:.4f}")
            logging.info(f"Best score so far: {self.best_score:.4f}")
            
            # Update policy when enough experiences collected
            if len(self.experience_buffer.buffer) >= self.batch_size:
                self._update_policy()
                self.experience_buffer.clear()
            
            if (iteration + 1) % 10 == 0:
                self._save_results()
        
        self._save_results()
        return self.best_architecture
    
    def _create_action_dims(self):
        """Convert search space to action dimensions"""
        action_dims = {}
        for param, values in self.search_space.items():
            if isinstance(values, tuple):
                num_bins = 10
                action_dims[param] = num_bins
            else:
                action_dims[param] = len(values)
        return action_dims
    
    def _get_current_state(self):
        """Create state vector from current architecture and constraints"""
        arch_state = []
        for param, values in self.search_space.items():
            if isinstance(values, tuple):
                # Normalize continuous values
                normalized = (self.best_architecture[param] - values[0]) / (values[1] - values[0])
                arch_state.append(normalized)
            else:
                # One-hot encode
                idx = values.index(self.best_architecture[param])
                normalized = idx / len(values)
                arch_state.append(normalized)
        
        hw_state = [
            self.hw_constraints['flash_size'] / (1024 * 1024),  # MB
            self.hw_constraints['ram_size'] / (1024 * 1024),    # MB
            self.hw_constraints['cpu_frequency'] / 1_000_000    # MHz
        ]
        
        return torch.tensor(arch_state + hw_state, dtype=torch.float32, device=self.device)
    
    def _sample_architecture(self, state):
        """Sample architecture parameters using current policy"""
        with torch.no_grad():
            action_probs = self.policy_net(state.unsqueeze(0))
        
        actions = {}
        log_probs = {}
        architecture = {}
        
        for param, probs in action_probs.items():
            # sample action using probs
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Convert action to architecture param
            if isinstance(self.search_space[param], tuple):
                min_val, max_val = self.search_space[param]
                value = min_val + (action.item() / (self.action_dims[param] - 1)) * (max_val - min_val)
                architecture[param] = round(value)
            else:
                architecture[param] = self.search_space[param][action.item()]
            
            actions[param] = action
            log_probs[param] = log_prob
        
        return architecture, log_probs
    
    def _architecture_to_actions(self, architecture):
        """Convert architecture parameters back to action indices"""
        actions = {}
        for param, value in architecture.items():
            if isinstance(self.search_space[param], tuple):
                min_val, max_val = self.search_space[param]
                action = int((value - min_val) / (max_val - min_val) * (self.action_dims[param] - 1))
            else:
                action = self.search_space[param].index(value)
            actions[param] = torch.tensor(action, device=self.device)
        return actions
    
    def _update_policy(self):
        """Update policy and value networks using PPO"""
        states, actions, rewards, old_log_probs = self.experience_buffer.get_batch()
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        for param in actions:
            actions[param] = actions[param].to(self.device)
            old_log_probs[param] = old_log_probs[param].to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            # curr action probs and values
            action_probs = self.policy_net(states)
            values = self.value_net(states).squeeze()
            advantages = rewards - values.detach()
            
            policy_loss = 0
            entropy_loss = 0
            for param in self.action_dims.keys():
                dist = torch.distributions.Categorical(action_probs[param])
                new_log_probs = dist.log_prob(actions[param])
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_log_probs - old_log_probs[param])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss -= torch.min(surr1, surr2).mean()
                entropy_loss -= entropy
            
            value_loss = F.mse_loss(values, rewards)
            
            # Total  loss
            total_loss = (
                policy_loss +
                self.value_coef * value_loss +
                self.entropy_coef * entropy_loss
            )
            
            # update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
    
    def _update_best_architectures(self, architecture, score, top_k=5):
        self.best_architectures.append({
            'architecture': architecture,
            'score': score,
            'timestamp': datetime.now().isoformat()
        })
        self.best_architectures.sort(key=lambda x: x['score'], reverse=True)
        self.best_architectures = self.best_architectures[:top_k]
    
    def _save_results(self):
        with open(self.output_dir / 'search_history.json', 'w') as f:
            json.dump(self.search_history, f, indent=2)
        
        # Top k best architectures
        with open(self.output_dir / 'best_architectures.json', 'w') as f:
            json.dump(self.best_architectures, f, indent=2)
        
        # save best architecture separately
        with open(self.output_dir / 'best_architecture.json', 'w') as f:
            json.dump({
                'architecture': self.best_architecture,
                'score': self.best_score,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

