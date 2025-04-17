search_space = {
    'num_layers': (1, 2),
    'hidden_size': [64, 128, 256],  # All divisible by max num_heads (4)
    'num_heads': [2, 4],
    'ff_dim': [128, 256],
    'vocab_size': [500, 1000],
    'quantization_bits': [4, 8],
}

# STM32F4 constraints
hw_constraints = {
    'flash_size': 1024 * 1024,      # 1MB Flash
    'ram_size': 192 * 1024,         # 192KB RAM
    'cpu_frequency': 168_000_000,   # 168MHz
}

