# RL-NAS-STM32

Neural Architecture Search (NAS) for optimizing tiny language models on STM32 microcontrollers. This project implements both Reinforcement Learning (RL) and Bayesian Optimization approaches for architecture search.

## Overview

This project automatically searches for optimal neural network architectures that:

- Fit within STM32F4 hardware constraints (Ex: 1MB Flash, 192KB RAM, 168MHz CPU)
- Perform character-level language modeling on tiny Shakespeare dataset
- Balance model accuracy with hardware efficiency

## Installation

```bash
git clone https://github.com/Veeraraju-E/RL-NAS-STM32.git
cd RL-NAS-STM32
pip install -r requirements.txt
```

## Usage

### Default Run

```bash
python nas/examples/run_search.py
```

This will use default settings:

- Bayesian optimization strategy
- 50 search iterations
- 5 training epochs per evaluation
- Batch size of 32

### Recommended Run Configurations

1. Quick Test Run (for testing setup):

```bash
python nas/examples/run_search.py --strategy bayesian --iterations 10 --epochs 2
```

2. Bayesian Optimization (good for preliminary search):

```bash
python nas/examples/run_search.py --strategy bayesian --iterations 100 --epochs 5
```

3. RL-based Search (for a more exploratory search):

```bash
python nas/examples/run_search.py --strategy rl --iterations 200 --epochs 5
```

### Args

- `--strategy`: Search strategy ['bayesian', 'rl'] (default: 'bayesian')
- `--epochs`: Number of training epochs per evaluation (default: 5)
- `--iterations`: Number of search iterations (default: 50)
- `--batch-size`: Training batch size (default: 32)

## Search Space

Example architecture search space includes:

```python
{
    'num_layers': (1, 2),
    'hidden_size': [64, 128, 256],
    'num_heads': [2, 4],
    'ff_dim': [128, 256],
    'vocab_size': [500, 1000],
    'quantization_bits': [4, 8],
}
```

- feel free to change this in the `congif/search_space.py` file

## Hardware Constraints

Targeting STM32F4 microcontroller:

- Flash Memory: 1MB
- RAM: 192KB
- CPU Frequency: 168MHz

## Scoring Metric

The architecture evaluation uses a combined scoring metric that balances multiple factors:

```python
score = (
    0.5 * accuracy +           # Model's prediction accuracy
    0.25 * size_score +        # Flash memory efficiency
    0.25 * speed_score         # Inference speed estimate
)
```

Where:

- `accuracy`: Next-character prediction accuracy (0-1)
- `size_score`: 1 - (model_size/flash_size)
- `speed_score`: 1 - (inference_time/100ms)

An architecture scores 0 if it exceeds hardware constraints:

- Model size > 1MB Flash
- RAM usage > 192KB
- Inference time > 100ms at 168MHz

## Output Structure

Each run creates a timestamped directory under `runs/` containing:

- `search.log`: Detailed search progress
- `config.json`: Search configuration
- `args.json`: Command line arguments
- `final_results.json`: Best architecture and scores

For RL search, additional files under `search_results/`:

- `search_history.json`: Complete search history
- `best_architectures.json`: Top-K architectures found
- `best_architecture.json`: Single best architecture

## Task Description

The model performs character-level language modeling on the tiny Shakespeare dataset:

- Input: Sequence of characters (length 32)
- Output: Prediction of next character for each position
- Vocabulary: ~65 unique characters (including letters, punctuation, whitespace)
- Evaluation: Based on next-character prediction accuracy and hardware constraints

## Contributing

1. Fork the repository
2. Make your changes
3. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
