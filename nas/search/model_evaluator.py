import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add the parent directory to system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from models.tiny_llm import TinyLLM
from data.tiny_eval_dataset import get_eval_dataloaders

class ModelEvaluator:
    def __init__(self, hardware_constraints, device, batch_size=32, epochs=3):
        self.hw_constraints = hardware_constraints
        self.device = device
        self.epochs = epochs
        (self.train_loader, self.val_loader), self.vocab_size = get_eval_dataloaders(batch_size=batch_size)
        
    def evaluate_architecture(self, architecture_params):
        """
        Evaluates architecture considering both hardware constraints and model performance
        Returns a score between 0 and 1
        """
        
        # 1. Check hardware constraints
        model_size = self._estimate_model_size(architecture_params)
        if model_size > self.hw_constraints['flash_size']:
            return 0.0
            
        # 2. Create and evaluate model
        try:
            model = TinyLLM(
                num_layers=int(architecture_params['num_layers']),
                hidden_size=int(architecture_params['hidden_size']),
                num_heads=int(architecture_params['num_heads']),
                ff_dim=int(architecture_params['ff_dim']),
                vocab_size=self.vocab_size,
                quantization_bits=int(architecture_params['quantization_bits'])
            ).to(self.device)
            
            
            accuracy = self._train_and_evaluate(model)
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return 0.0
            
        # 3. Hardware efficiency metrics
        flops = self._estimate_flops(architecture_params)
        inf_time = flops / self.hw_constraints['cpu_frequency']
        size_score = 1 - model_size/self.hw_constraints['flash_size']
        speed_score = 1 - inf_time/100
        
        # Combined score with accuracy
        score = (
            0.5 * accuracy +           # actual model performance
            0.25 * size_score +        # size efficiency
            0.25 * speed_score         # speed estimate
        )
        
        
        return score
        
    def _train_and_evaluate(self, model):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            total_loss = 0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                # Ensure input matches model's sequence length
                if x.size(1) > model.seq_length:
                    x = x[:, :model.seq_length]
                    y = y[:, :model.seq_length]
                elif x.size(1) < model.seq_length:
                    pad_size = model.seq_length - x.size(1)
                    x = torch.nn.functional.pad(x, (0, pad_size), value=0)
                    y = torch.nn.functional.pad(y, (0, pad_size), value=0)
                
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            # Evaluation phase
            model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    # Apply same sequence length handling for validation
                    if x.size(1) > model.seq_length:
                        x = x[:, :model.seq_length]
                        y = y[:, :model.seq_length]
                    elif x.size(1) < model.seq_length:
                        pad_size = model.seq_length - x.size(1)
                        x = torch.nn.functional.pad(x, (0, pad_size), value=0)
                        y = torch.nn.functional.pad(y, (0, pad_size), value=0)
                    
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    val_loss += F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1)).item()
                    _, predicted = output.max(-1)
                    total += y.size(0) * y.size(1)
                    correct += (predicted == y).sum().item()
            
            accuracy = correct / total
            best_accuracy = max(best_accuracy, accuracy)
            print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}')
        
        return best_accuracy

    def _estimate_flops(self, params):
        """Estimate computational complexity"""
        # Transformer FLOPs estimation
        seq_length = 32  # from model config
        h = params['hidden_size']
        l = params['num_layers']
        ff_dim = params['ff_dim']
        
        # Attention FLOPs per layer
        attn_flops = seq_length * seq_length * h * 2
        
        # FFN FLOPs per layer
        ffn_flops = seq_length * h * ff_dim * 2
        
        total_flops = (attn_flops + ffn_flops) * l
        return total_flops
        
    def _estimate_model_size(self, params):
        """Estimate model size in bytes"""
        h = params['hidden_size']
        l = params['num_layers']
        ff_dim = params['ff_dim']
        v = params['vocab_size']
        bits = params['quantization_bits']
        
        # Parameter count estimation
        embedding_params = v * h
        transformer_params = l * (
            4 * h * h +  # attention matrices
            2 * h * ff_dim  # feedforward layers
        )
        
        total_params = embedding_params + transformer_params
        size_bytes = (total_params * bits) / 8
        return size_bytes