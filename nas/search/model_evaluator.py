import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add the parent directory to system path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from models.tiny_llm import TinyLLM
from data.tiny_eval_dataset import get_eval_dataloaders

class ModelEvaluator:
    def __init__(self, hardware_constraints, batch_size=32):
        self.hw_constraints = hardware_constraints
        (self.train_loader, self.val_loader), self.vocab_size = get_eval_dataloaders(batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_architecture(self, architecture_params):
        """
        Evaluates a model architecture and returns a score between 0 and 1
        """
        print("\nEvaluating architecture:", architecture_params)
        
        # Quick size estimation before creating model
        estimated_size = (
            architecture_params['num_layers'] *
            architecture_params['hidden_size'] *
            architecture_params['ff_dim'] *
            4  # Approximate bytes per parameter
        )
        
        if estimated_size > self.hw_constraints['flash_size']:
            print(f"Estimated size {estimated_size/1024/1024:.2f}MB exceeds flash constraint")
            return 0.0
        
        # Create model with given parameters
        model = TinyLLM(
            num_layers=int(architecture_params['num_layers']),
            hidden_size=int(architecture_params['hidden_size']),
            num_heads=int(architecture_params['num_heads']),
            ff_dim=int(architecture_params['ff_dim']),
            vocab_size=self.vocab_size,  # Use vocab size from dataset
            quantization_bits=int(architecture_params['quantization_bits'])
        )
        
        # Check hardware constraints
        model_size = self._estimate_model_size(model)
        print(f"Model size: {model_size/1024/1024:.2f}MB")
        print(f"Flash constraint: {self.hw_constraints['flash_size']/1024/1024:.2f}MB")
        
        if model_size > self.hw_constraints['flash_size']:
            print("❌ Model rejected: Exceeds flash size constraint")
            return 0.0
        
        # Train model for a few epochs
        print("Training model...")
        accuracy = self._quick_train_and_evaluate(model)
        print(f"Training accuracy: {accuracy:.4f}")
        
        # Estimate inference time
        inf_time = self._estimate_inference_time(model)
        print(f"Estimated inference time: {inf_time:.2f}ms")
        
        # Combined score (example weighting)
        score = 0.6 * accuracy + 0.2 * (1 - model_size/self.hw_constraints['flash_size']) + \
                0.2 * (1 - inf_time/100)  # assuming 100ms is max acceptable
        
        print(f"Final score components:")
        print(f"- Accuracy contribution: {0.6 * accuracy:.4f}")
        print(f"- Size contribution: {0.2 * (1 - model_size/self.hw_constraints['flash_size']):.4f}")
        print(f"- Speed contribution: {0.2 * (1 - inf_time/100):.4f}")
        print(f"Total score: {score:.4f}")
        
        return score
    
    def _estimate_model_size(self, model):
        """Estimate model size in bytes"""
        param_size = 0
        for param in model.parameters():
            # Account for quantization
            bits_per_param = 4 if model.quantization_bits == 4 else 8
            param_size += param.nelement() * bits_per_param / 8
        
        # Add 10% overhead for model metadata
        total_size = param_size * 1.1
        return total_size
    
    def _quick_train_and_evaluate(self, model, epochs=1):
        """Quick training to get rough performance estimate"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        model.train()
        for epoch in range(epochs):
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)  # Shape: [batch, seq_len, vocab_size]
                
                # Reshape for cross entropy
                outputs = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
                batch_labels = batch_labels.view(-1)  # [batch*seq_len]
                
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in self.val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_data)  # [batch, seq_len, vocab_size]
                outputs = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
                batch_labels = batch_labels.view(-1)  # [batch*seq_len]
                
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        
        return correct / total
    
    def _estimate_inference_time(self, model):
        """Estimate inference time in milliseconds"""
        # Implement inference time estimation
        return 50  # placeholder







