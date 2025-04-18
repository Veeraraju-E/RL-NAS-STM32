import numpy as np
from collections import deque
import torch

class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state, actions, reward, log_probs):
        self.buffer.append((state, actions, reward, log_probs))
        
    def clear(self):
        self.buffer.clear()
        
    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.stack([b[0] for b in batch])
        actions = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1].keys()}
        rewards = torch.tensor([b[2] for b in batch])
        log_probs = {k: torch.stack([b[3][k] for b in batch]) for k in batch[0][3].keys()}
        
        return states, actions, rewards, log_probs