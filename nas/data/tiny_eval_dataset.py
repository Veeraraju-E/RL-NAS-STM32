import torch
from torch.utils.data import Dataset, DataLoader
import requests
import os
from pathlib import Path

class TinyTextDataset(Dataset):
    def __init__(self, seq_length=32, split='train'):
        self.seq_length = seq_length
        
        # Download tiny dataset if not exists
        self.data_path = self._download_dataset()
        
        # Load and preprocess data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create vocabulary (simple character-level for demo)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
        # Split data
        train_size = int(0.8 * len(self.data))
        if split == 'train':
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size:]
            
    def __len__(self):
        return max(0, len(self.data) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y
    
    def _download_dataset(self):
        data_dir = Path(__file__).parent / 'data'
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir / 'tiny_shakespeare.txt'
        
        if not data_path.exists():
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print("Downloading tiny dataset...")
            response = requests.get(url)
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download complete!")
            
        return data_path

def get_eval_dataloaders(batch_size=32):
    """Create train/test splits of the tiny text dataset"""
    train_dataset = TinyTextDataset(split='train')
    test_dataset = TinyTextDataset(split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return (train_loader, test_loader), train_dataset.vocab_size
