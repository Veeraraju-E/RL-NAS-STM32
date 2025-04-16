SEARCH_SPACE = {
    'num_layers': (1, 2),           # Reduced from (1, 4)
    'hidden_size': [32, 64, 128],   # Added smaller size, removed 256
    'num_heads': [2, 4],            # Removed 8 heads option
    'ff_dim': [64, 128, 256],       # Added smaller size, removed 512
    'vocab_size': [500, 1000],      # Reduced from [1000, 2000]
    'quantization_bits': [4, 8],    # Kept same
}

# STM32F4 constraints
HARDWARE_CONSTRAINTS = {
    'flash_size': 1024 * 1024,      # 1MB Flash
    'ram_size': 192 * 1024,         # 192KB RAM
    'cpu_frequency': 168_000_000,   # 168MHz
}
