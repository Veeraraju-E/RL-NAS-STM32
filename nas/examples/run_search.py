import sys
from pathlib import Path
import torch
import argparse
import logging

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config.search_space import search_space, hw_constraints
from search.bayesian_optimizer import BayesianSearcher
from search.model_evaluator import ModelEvaluator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_hardware_info():
    logging.info("\nHardware Constraints:")
    logging.info(f"Flash Size: {hw_constraints['flash_size']/1024/1024:.2f}MB")
    logging.info(f"RAM Size: {hw_constraints['ram_size']/1024:.2f}KB")
    logging.info(f"CPU Frequency: {hw_constraints['cpu_frequency']/1_000_000:.0f}MHz")
    logging.info(f"Using Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

def main():
    setup_logging()
    print_hardware_info()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Run architecture search')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--iterations', type=int, default=50, help='Number of search iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--project', type=str, default='tiny-llm-nas', help='WandB project name')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity/username')
    args = parser.parse_args()


    evaluator = ModelEvaluator(
        hardware_constraints=hw_constraints,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    
    searcher = BayesianSearcher(
        search_space=search_space,
        hardware_constraints=hw_constraints,
        evaluator=evaluator,
        device=device
    )
    
    logging.info("Starting architecture search...")
    logging.info(f"Search space: {search_space}")
    
    try:
        best_architecture = searcher.search(num_iterations=args.iterations)
        
        logging.info("\nSearch completed!")
        logging.info(f"Best architecture found: {best_architecture}")
        logging.info(f"Best score: {searcher.best_score:.4f}")
        
        if best_architecture:
            logging.info("\nPerforming final evaluation of best architecture:")
            final_score = evaluator.evaluate_architecture(best_architecture)
            logging.info(f"Final evaluation score: {final_score:.4f}")
            
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        raise

if __name__ == "__main__":
    main()
