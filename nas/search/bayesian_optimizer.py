import sys
from pathlib import Path
import torch
from bayes_opt import BayesianOptimization

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from search.base_searcher import BaseSearcher


class BayesianSearcher(BaseSearcher):
    def __init__(self, search_space, hardware_constraints, evaluator, device):
        super().__init__(search_space, hardware_constraints)
        self.evaluator = evaluator
        self.device = device
        self.param_bounds = self._create_bounds()
        
    def search(self, num_iterations):
        """Execute Bayesian optimization search"""
        optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds=self.param_bounds,
            verbose=2,
            random_state=42
        )
        
        optimizer.set_gp_params(kernel=None, alpha=1e-6)
        
        init_points = min(5, num_iterations // 2)
        remaining_iter = num_iterations - init_points
        
        print(f"\nInitializing with {init_points} random points...")
        optimizer.maximize(
            init_points=init_points,
            n_iter=0,
        )
        
        print(f"\nRunning Bayesian optimization for {remaining_iter} iterations...")
        optimizer.maximize(
            init_points=0,
            n_iter=remaining_iter,
        )
        
        best_params = optimizer.max['params']
        self.best_score = optimizer.max['target']
        self.best_architecture = self._convert_to_arch_params(best_params)
        
        return self.best_architecture
    
    def _objective_function(self, **params):
        """Objective function for Bayesian optimization"""
        try:
            arch_params = self._convert_to_arch_params(params)
            
            # Validate architecture parameters
            if not self._validate_architecture(arch_params):
                return 0.0
            
            arch_params = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in arch_params.items()
            }
            
            score = self.evaluator.evaluate_architecture(arch_params)
            return score
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 0.0
    
    def _convert_to_arch_params(self, params):
        """Convert optimization parameters to architecture parameters"""
        arch_params = {}
        for param_name, value in params.items():
            if param_name in ['num_layers', 'hidden_size', 'num_heads', 
                            'ff_dim', 'vocab_size', 'quantization_bits']:
                # Round to nearest valid value
                if param_name == 'hidden_size':
                    # Ensure hidden_size is divisible by maximum num_heads
                    max_heads = max(self.search_space['num_heads'])
                    value = round(value / max_heads) * max_heads
                elif param_name == 'num_heads':
                    value = round(value)
                    # Ensure num_heads divides hidden_size
                    hidden_size = round(params['hidden_size'])
                    value = max(1, min(value, hidden_size))
                else:
                    value = round(value)
                arch_params[param_name] = value
            else:
                arch_params[param_name] = value
        return arch_params
    
    def _validate_architecture(self, arch_params):
        """Validate architecture parameters"""
        try:
            hidden_size = arch_params['hidden_size']
            num_heads = arch_params['num_heads']
            return (hidden_size % num_heads == 0 and
                   num_heads > 0 and
                   hidden_size > 0)
        except Exception:
            return False
    
    def _create_bounds(self):
        """Convert search space to bayes_opt format"""
        bounds = {}
        for param, values in self.search_space.items():
            if isinstance(values, tuple):
                bounds[param] = values
            else:
                bounds[param] = (min(values), max(values))
        return bounds