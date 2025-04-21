import sys
from pathlib import Path
import torch
import argparse
import logging
import json
from datetime import datetime

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config.search_space import search_space, hw_constraints
from search.bayesian_optimizer import BayesianSearcher
from search.rl_searcher import RLSearcher
from search.model_evaluator import ModelEvaluator

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / 'search.log')
    console_handler = logging.StreamHandler()
    
    # fix format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.root.handlers = []
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.INFO)

def print_hardware_info():
    logging.info("\nHardware Constraints:")
    logging.info(f"Flash Size: {hw_constraints['flash_size']/1024/1024:.2f}MB")
    logging.info(f"RAM Size: {hw_constraints['ram_size']/1024:.2f}KB")
    logging.info(f"CPU Frequency: {hw_constraints['cpu_frequency']/1_000_000:.0f}MHz")
    logging.info(f"Using Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

def create_searcher(strategy, search_space, hw_constraints, evaluator, device):
    strategies = {'bayesian': BayesianSearcher, 'rl': RLSearcher}
    
    if strategy not in strategies:
        raise ValueError(f"Unknown search strategy: {strategy}. Available strategies: {list(strategies.keys())}")
    
    return strategies[strategy](
        search_space=search_space,
        hardware_constraints=hw_constraints,
        evaluator=evaluator,
        device=device
    )

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / timestamp
    setup_logging(output_dir)
    
    # Save config
    config = {
        'timestamp': timestamp,
        'search_space': search_space,
        'hardware_constraints': hw_constraints
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print_hardware_info()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Run architecture search')
    parser.add_argument('--strategy', type=str, default='bayesian', choices=['bayesian', 'rl'], help='Search strategy to use (bayesian or rl)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--iterations', type=int, default=50, help='Number of search iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    evaluator = ModelEvaluator(
        hardware_constraints=hw_constraints,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    
    searcher = create_searcher(
        args.strategy,
        search_space=search_space,
        hw_constraints=hw_constraints,
        evaluator=evaluator,
        device=device
    )
    
    logging.info(f"Starting architecture search using {args.strategy} strategy...")
    logging.info(f"Search space: {json.dumps(search_space, indent=2)}")
    
    try:
        best_architecture = searcher.search(num_iterations=args.iterations)
        
        logging.info("\nSearch completed!")
        logging.info(f"Best architecture found: {json.dumps(best_architecture, indent=2)}")
        logging.info(f"Best score: {searcher.best_score:.4f}")
        
        if best_architecture:
            logging.info("\nPerforming final evaluation of best architecture:")
            final_score = evaluator.evaluate_architecture(best_architecture)
            logging.info(f"Final evaluation score: {final_score:.4f}")
            
            final_results = {
                'best_architecture': best_architecture,
                'best_score': searcher.best_score,
                'final_evaluation_score': final_score
            }
            with open(output_dir / 'final_results.json', 'w') as f:
                json.dump(final_results, f, indent=2)
            
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        raise

if __name__ == "__main__":
    main()
