from abc import ABC, abstractmethod

class BaseSearcher(ABC):
    def __init__(self, search_space, hardware_constraints):
        self.search_space = search_space
        self.hw_constraints = hardware_constraints
        self.best_architecture = None
        self.best_score = float('-inf')
    
    @abstractmethod
    def search(self, num_iterations):
        pass
    
    def evaluate_architecture(self, architecture):
        """
        Evaluate architecture based on:
        1. Model accuracy
        2. Memory usage
        3. Inference time
        """
        if not self._meets_hardware_constraints(architecture):
            return float('-inf')
            
        # TODO: Implement actual evaluation
        return 0.0
    
    def _meets_hardware_constraints(self, architecture):
        # Estimate memory usage
        model_size = self._estimate_model_size(architecture)
        return (model_size < self.hw_constraints['flash_size'] and
                self._estimate_ram_usage(architecture) < self.hw_constraints['ram_size'])