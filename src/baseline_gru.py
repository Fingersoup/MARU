"""
Baseline GRU model for MARU project.

This module implements a simple baseline GRU model for comparison with the MARU architecture.
The model uses PyTorch's built-in GRU module for sequence processing.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaselineGRU(nn.Module):
    """
    A simple baseline GRU model for sequence processing.
    
    This model serves as a baseline for comparison with the MARU architecture.
    It uses PyTorch's built-in GRU module followed by a linear output layer.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1  # For position prediction
    ):
        """
        Initialize the BaselineGRU model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Hidden dimension of the GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
            output_dim: Output dimension (1 for position prediction)
        """
        super(BaselineGRU, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output head for position prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            hidden: Initial hidden state (optional)
            
        Returns:
            output: Model output of shape (batch_size, output_dim)
            hidden: Final hidden state
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed input tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Pass through GRU
        gru_output, hidden = self.gru(embedded, hidden)  # (batch_size, seq_len, hidden_dim)
        
        # Use the final timestep output for prediction
        final_output = gru_output[:, -1, :]  # (batch_size, hidden_dim)
        
        # Pass through output head
        output = self.output_head(final_output)  # (batch_size, output_dim)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Initial hidden state
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


def create_baseline_gru(vocab_size: int, **kwargs) -> BaselineGRU:
    """
    Factory function to create a BaselineGRU model.
    
    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional arguments for BaselineGRU
        
    Returns:
        BaselineGRU model instance
    """
    return BaselineGRU(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Simple test
    vocab_size = 100
    batch_size = 4
    seq_len = 50
    
    model = create_baseline_gru(vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output, hidden = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print("BaselineGRU test passed!")
