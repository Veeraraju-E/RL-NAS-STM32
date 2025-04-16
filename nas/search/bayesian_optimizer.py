import sys
from pathlib import Path

# Add the parent directory to system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from search.base_searcher import BaseSearcher
import GPyOpt


class BayesianSearcher(BaseSearcher):
    def __init__(self, search_space, hardware_constraints, evaluator):
        super().__init__(search_space, hardware_constraints)
        self.bounds = self._create_bounds()
        self.evaluator = evaluator
        
    def search(self, num_iterations):
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=self._objective_function,
            domain=self.bounds,
            model_type='GP',
            acquisition_type='EI',
            maximize=True,
            normalize_Y=True
        )
        
        optimizer.run_optimization(max_iter=num_iterations)
        self.best_architecture = self._convert_to_arch_params(optimizer.x_opt)
        self.best_score = optimizer.fx_opt
        return self.best_architecture
    
    def _objective_function(self, x):
        """Convert GPyOpt parameters to architecture and evaluate"""
        arch_params = self._convert_to_arch_params(x[0])
        score = self.evaluator.evaluate_architecture(arch_params)
        return -score  # GPyOpt minimizes, so we negate the score
    
    def _convert_to_arch_params(self, x):
        """Convert optimization parameters to architecture parameters"""
        params = {}
        for i, (param_name, values) in enumerate(self.search_space.items()):
            # Round continuous parameters that must be integers
            if param_name in ['num_layers', 'hidden_size', 'num_heads', 'ff_dim', 'vocab_size', 'quantization_bits']:
                params[param_name] = round(x[i])
            else:
                params[param_name] = x[i]
        return params
        
    def _create_bounds(self):
        # Convert search space to GPyOpt format
        bounds = []
        for param, values in self.search_space.items():
            if isinstance(values, tuple):
                bounds.append({'name': param, 'type': 'continuous',
                             'domain': values})
            else:
                bounds.append({'name': param, 'type': 'discrete',
                             'domain': values})
        return bounds

