search_space = {
    'num_layers': (1, 2),
    'hidden_size': [64, 128, 256],
    'num_heads': [2, 4],
    'ff_dim': [128, 256],
    'vocab_size': [500, 1000],
    'quantization_bits': [4, 8],
}

# Hardware constraints
hw_constraints = {
    'flash_size': 128 * 1024,      # 128 KB Flash
    'ram_size': 1 * 1024 * 1024,       # 1MB RAM
    'cpu_frequency': 180_000_000,       # 180MHz
}
