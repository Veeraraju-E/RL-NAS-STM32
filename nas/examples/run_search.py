import sys
from pathlib import Path
import torch

# Add the parent directory to system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config.search_space import SEARCH_SPACE, HARDWARE_CONSTRAINTS
from search.bayesian_optimizer import BayesianSearcher
from search.model_evaluator import ModelEvaluator

def print_hardware_info():
    print("\nHardware Constraints:")
    print(f"Flash Size: {HARDWARE_CONSTRAINTS['flash_size']/1024/1024:.2f}MB")
    print(f"RAM Size: {HARDWARE_CONSTRAINTS['ram_size']/1024:.2f}KB")
    print(f"CPU Frequency: {HARDWARE_CONSTRAINTS['cpu_frequency']/1_000_000:.0f}MHz")
    print(f"Using Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

def main():
    print_hardware_info()
    
    # Create evaluator with tiny dataset
    evaluator = ModelEvaluator(HARDWARE_CONSTRAINTS, batch_size=32)
    
    # Create and run searcher
    searcher = BayesianSearcher(SEARCH_SPACE, HARDWARE_CONSTRAINTS, evaluator)
    
    print("Starting architecture search...")
    print("Search space:", SEARCH_SPACE)
    
    best_architecture = searcher.search(num_iterations=50)
    
    print("\nSearch completed!")
    print("Best architecture found:", best_architecture)
    print("Best score:", searcher.best_score)
    
    # Create and evaluate best model to show detailed metrics
    if best_architecture:
        print("\nEvaluating best architecture:")
        final_score = evaluator.evaluate_architecture(best_architecture)
        print(f"Final evaluation score: {final_score:.4f}")

if __name__ == "__main__":
    main()

