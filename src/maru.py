"""
MARU (Memory-Augmented Recurrent Unit) Architecture.

This module implements the complete MARU architecture that combines:
1. Input embedding layer
2. MambaBlock for initial sequence processing with selective state-space modeling
3. MoMGRUCell for recurrent processing with memory augmentation
4. Output head for final predictions

The architecture processes sequences in two stages:
- Stage 1: MambaBlock processes the entire sequence in parallel, capturing long-range dependencies
- Stage 2: MoMGRUCell processes the Mamba output recurrently, maintaining persistent memory

This design achieves O(1) inference complexity for autoregressive generation while maintaining
rich contextual understanding through the memory mechanisms.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
import math
import logging

from mamba_block import MambaBlock, MambaConfig
from mom_gru_cell import MoMGRUCell
from enhanced_mom_gru_cell import EnhancedMoMGRUCell
from enhanced_mom_gru_config import EnhancedMoMGRUConfig, get_conservative_config

logger = logging.getLogger(__name__)


class MARUConfig:
    """Configuration class for MARU architecture."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        embedding_dim: Optional[int] = None,
        hidden_size: int = 256,
        output_dim: int = 1,
        # Mamba configuration
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        # Memory configuration
        memory_size: int = 128,
        memory_dim: int = 64,
        num_memories: int = 4,
        router_hidden_dim: int = 32,
        # Enhanced MoM-GRU configuration
        use_enhanced_mom_gru: bool = True,
        enhanced_mom_gru_config: Optional[EnhancedMoMGRUConfig] = None,
        # Training configuration
        dropout: float = 0.1,
        bias: bool = True,
        # Device configuration
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize MARU configuration.

        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension for Mamba block
            embedding_dim: Embedding dimension (defaults to d_model if None)
            hidden_size: Hidden size for GRU cell
            output_dim: Output dimension for final prediction
            mamba_d_state: State dimension for Mamba
            mamba_d_conv: Convolution dimension for Mamba
            mamba_expand: Expansion factor for Mamba
            memory_size: Number of slots in each memory matrix
            memory_dim: Dimension of each memory slot
            num_memories: Number of separate memory matrices
            router_hidden_dim: Hidden dimension for memory router
            use_enhanced_mom_gru: Whether to use Enhanced MoM-GRU Cell
            enhanced_mom_gru_config: Configuration for Enhanced MoM-GRU features
            dropout: Dropout probability
            bias: Whether to use bias terms
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_dim = embedding_dim or d_model
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # Mamba configuration
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        # Memory configuration
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_memories = num_memories
        self.router_hidden_dim = router_hidden_dim

        # Enhanced MoM-GRU configuration
        self.use_enhanced_mom_gru = use_enhanced_mom_gru
        if enhanced_mom_gru_config is None and use_enhanced_mom_gru:
            # Use conservative config by default
            self.enhanced_mom_gru_config = get_conservative_config()
            # Set device and dtype to match MARU config
            self.enhanced_mom_gru_config.device = device
            self.enhanced_mom_gru_config.dtype = dtype
        else:
            self.enhanced_mom_gru_config = enhanced_mom_gru_config



        # Training configuration
        self.dropout = dropout
        self.bias = bias

        # Device configuration
        self.device = device
        self.dtype = dtype


class MARU(nn.Module):
    """
    Memory-Augmented Recurrent Unit (MARU) Architecture.
    
    MARU combines the strengths of state-space models (Mamba) and memory-augmented
    recurrent networks to achieve efficient long-sequence modeling with O(1) inference.
    """
    
    def __init__(self, config: MARUConfig):
        """
        Initialize the MARU model.
        
        Args:
            config: MARU configuration object
        """
        super().__init__()
        self.config = config
        
        # Input embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embedding_dim,
            device=config.device,
            dtype=config.dtype
        )
        
        # Projection layer if embedding_dim != d_model
        if config.embedding_dim != config.d_model:
            self.embed_proj = nn.Linear(
                config.embedding_dim, 
                config.d_model,
                bias=config.bias,
                device=config.device,
                dtype=config.dtype
            )
        else:
            self.embed_proj = nn.Identity()
        
        # Mamba block for initial sequence processing
        mamba_config = MambaConfig(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            bias=config.bias
        )
        self.mamba_block = MambaBlock(mamba_config)
        
        # Projection layer from Mamba output to GRU input
        self.mamba_to_gru_proj = nn.Linear(
            config.d_model,
            config.hidden_size,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype
        )
        
        # Memory-augmented GRU cell (Enhanced or Original)
        if config.use_enhanced_mom_gru:
            self.mom_gru_cell = EnhancedMoMGRUCell(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                memory_dim=config.memory_dim,
                num_memories=config.num_memories,
                router_hidden_dim=config.router_hidden_dim,
                bias=config.bias,
                device=config.device,
                dtype=config.dtype,
                config=config.enhanced_mom_gru_config
            )
        else:
            self.mom_gru_cell = MoMGRUCell(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                memory_dim=config.memory_dim,
                num_memories=config.num_memories,
                router_hidden_dim=config.router_hidden_dim,
                bias=config.bias,
                device=config.device,
                dtype=config.dtype
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Output head for final prediction
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=config.bias),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_dim, bias=config.bias)
        )
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        # Initialize projection layers
        for module in [self.embed_proj, self.mamba_to_gru_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output head
        for module in self.output_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the MARU model.

        The forward pass consists of:
        1. Embed input token IDs
        2. Process entire sequence through MambaBlock (parallel processing)
        3. Process Mamba output recurrently through MoMGRUCell (sequential processing)
        4. Use final hidden state for prediction

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            hidden: Initial hidden state for GRU cell (optional)
            memory_state: Initial memory state for GRU cell (optional)
            return_hidden: Whether to return the final hidden state

        Returns:
            output: Model prediction of shape (batch_size, output_dim)
            hidden: Final hidden state (if return_hidden=True)
        """
        batch_size, seq_len = input_ids.shape

        # Stage 1: Embedding and Mamba processing
        # Embed input tokens
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embed_proj(embedded)  # (batch_size, seq_len, d_model)
        embedded = self.dropout(embedded)

        # Process through Mamba block (parallel processing of entire sequence)
        mamba_output = self.mamba_block(embedded)  # (batch_size, seq_len, d_model)

        # Project Mamba output to GRU input dimension
        gru_input = self.mamba_to_gru_proj(mamba_output)  # (batch_size, seq_len, hidden_size)
        gru_input = self.dropout(gru_input)

        # Stage 2: Recurrent processing through MoM-GRU
        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.mom_gru_cell.get_initial_memory_state(batch_size)

        # Process sequence step by step through the memory-augmented GRU cell
        # Collect hidden states at each timestep for language modeling
        current_hidden = hidden
        current_memory_state = memory_state
        all_hidden_states = []

        for t in range(seq_len):
            input_step = gru_input[:, t, :]  # (batch_size, hidden_size)
            current_hidden, _, current_memory_state = self.mom_gru_cell(
                input_step, current_hidden, current_memory_state
            )
            all_hidden_states.append(current_hidden)

        # Stage 3: Predictions for all timesteps (for language modeling)
        # Stack all hidden states and generate predictions for each timestep
        all_hidden = torch.stack(all_hidden_states, dim=1)  # (batch_size, seq_len, hidden_size)

        # Apply output head to all timesteps
        batch_size, seq_len, hidden_size = all_hidden.shape
        all_hidden_flat = all_hidden.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        output_flat = self.output_head(all_hidden_flat)  # (batch_size * seq_len, output_dim)
        output = output_flat.view(batch_size, seq_len, -1)  # (batch_size, seq_len, output_dim)

        final_hidden = current_hidden  # Keep final hidden state for compatibility

        if return_hidden:
            return output, (final_hidden, current_memory_state)
        else:
            return output, None

    def generate_step(
        self,
        input_id: torch.Tensor,
        hidden: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a single step for autoregressive generation.

        This method is optimized for O(1) inference during generation.
        It processes a single token through the model and returns the
        updated hidden state and memory state for the next step.

        Args:
            input_id: Single input token ID of shape (batch_size, 1)
            hidden: Current hidden state of shape (batch_size, hidden_size)
            memory_state: Current memory state (optional)

        Returns:
            output: Model prediction of shape (batch_size, output_dim)
            new_hidden: Updated hidden state of shape (batch_size, hidden_size)
            new_memory_state: Updated memory state
        """
        batch_size = input_id.shape[0]

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.mom_gru_cell.get_initial_memory_state(batch_size)

        # Embed the single token
        embedded = self.embedding(input_id)  # (batch_size, 1, embedding_dim)
        embedded = self.embed_proj(embedded)  # (batch_size, 1, d_model)
        embedded = self.dropout(embedded)

        # Process through Mamba block
        mamba_output = self.mamba_block(embedded)  # (batch_size, 1, d_model)

        # Project to GRU input dimension
        gru_input = self.mamba_to_gru_proj(mamba_output)  # (batch_size, 1, hidden_size)
        gru_input = self.dropout(gru_input)

        # Process single step through MoM-GRU
        input_step = gru_input.squeeze(1)  # (batch_size, hidden_size)
        new_hidden, _, new_memory_state = self.mom_gru_cell(input_step, hidden, memory_state)

        # Generate prediction
        output = self.output_head(new_hidden)  # (batch_size, output_dim)

        return output, new_hidden, new_memory_state

    def get_initial_memory_state(self, batch_size: int) -> torch.Tensor:
        """Get initial memory state for the given batch size."""
        return self.mom_gru_cell.get_initial_memory_state(batch_size)

    def reset_memory_parameters(self):
        """Reset the memory parameters of the MoM-GRU cell."""
        self.mom_gru_cell.reset_memory_parameters()

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get a breakdown of parameters by component."""
        breakdown = {}

        breakdown['embedding'] = sum(p.numel() for p in self.embedding.parameters())
        breakdown['embed_proj'] = sum(p.numel() for p in self.embed_proj.parameters()) if hasattr(self.embed_proj, 'parameters') else 0
        breakdown['mamba_block'] = sum(p.numel() for p in self.mamba_block.parameters())
        breakdown['mamba_to_gru_proj'] = sum(p.numel() for p in self.mamba_to_gru_proj.parameters())
        breakdown['mom_gru_cell'] = sum(p.numel() for p in self.mom_gru_cell.parameters())
        breakdown['output_head'] = sum(p.numel() for p in self.output_head.parameters())

        breakdown['total'] = sum(breakdown.values())

        return breakdown



    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics from Enhanced MoM-GRU.

        Returns:
            Dictionary of monitoring statistics
        """
        if hasattr(self.mom_gru_cell, 'get_monitoring_stats'):
            return self.mom_gru_cell.get_monitoring_stats()
        else:
            return {}








def create_maru_model(
    vocab_size: int,
    d_model: int = 256,
    hidden_size: int = 256,
    output_dim: int = 1,
    **kwargs
) -> MARU:
    """
    Factory function to create a MARU model with default configuration.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        hidden_size: Hidden size for GRU
        output_dim: Output dimension
        **kwargs: Additional configuration parameters

    Returns:
        MARU model instance
    """
    config = MARUConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        hidden_size=hidden_size,
        output_dim=output_dim,
        **kwargs
    )
    return MARU(config)



