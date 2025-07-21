"""
Memory-augmented GRU Cell (MoM-GRU) for the MARU project.

This module implements a GRU cell with Mixture-of-Memories (MoM) capability.
Unlike standard GRUs, this cell processes one timestep at a time and maintains
multiple memory matrices that can be selectively read from and written to.

The cell combines:
1. Standard GRU gating mechanisms (reset and update gates)
2. Multiple memory matrices for persistent storage
3. A router network for memory selection
4. Read/write heads for memory operations

Based on the GRU implementation from d2l.ai and extended with memory mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class MoMGRUCell(nn.Module):
    """
    Memory-augmented GRU Cell with Mixture-of-Memories.
    
    This cell extends the standard GRU with multiple memory matrices that can be
    selectively accessed based on the current input and hidden state. It processes
    one timestep at a time and maintains persistent memory across timesteps.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_size: int = 128,
        memory_dim: int = 64,
        num_memories: int = 4,
        router_hidden_dim: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the MoM-GRU Cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            memory_size: Number of slots in each memory matrix
            memory_dim: Dimension of each memory slot
            num_memories: Number of separate memory matrices
            router_hidden_dim: Hidden dimension for the router network
            bias: Whether to use bias in linear layers
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_memories = num_memories
        self.router_hidden_dim = router_hidden_dim

        # Initialize memory matrices as learnable parameters (static initialization)
        # These will be used as the base memory state
        self.initial_memories = nn.Parameter(
            torch.randn(num_memories, memory_size, memory_dim, **factory_kwargs) * 0.1
        )
        
        # GRU gate parameters (following d2l.ai implementation)
        # Reset gate parameters
        self.W_xr = nn.Parameter(torch.randn(input_size + memory_dim, hidden_size, **factory_kwargs))
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size, **factory_kwargs))
        self.b_r = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs)) if bias else None
        
        # Update gate parameters  
        self.W_xz = nn.Parameter(torch.randn(input_size + memory_dim, hidden_size, **factory_kwargs))
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size, **factory_kwargs))
        self.b_z = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs)) if bias else None
        
        # Candidate hidden state parameters
        self.W_xh = nn.Parameter(torch.randn(input_size + memory_dim, hidden_size, **factory_kwargs))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size, **factory_kwargs))
        self.b_h = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs)) if bias else None
        
        # Router network for memory selection
        # Takes current input and hidden state, outputs distribution over memories
        self.router = nn.Sequential(
            nn.Linear(input_size + hidden_size, router_hidden_dim, bias=bias, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_memories, bias=bias, **factory_kwargs)
        )
        
        # Read head parameters
        self.read_query = nn.Linear(input_size + hidden_size, memory_dim, bias=bias, **factory_kwargs)
        
        # Write head parameters  
        self.write_key = nn.Linear(hidden_size, memory_dim, bias=bias, **factory_kwargs)
        self.write_value = nn.Linear(hidden_size, memory_dim, bias=bias, **factory_kwargs)
        self.write_gate = nn.Linear(hidden_size, 1, bias=bias, **factory_kwargs)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization."""
        # Initialize GRU weights
        for weight in [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]:
            nn.init.xavier_uniform_(weight)
        
        # Initialize router and memory operation weights
        for module in [self.router, self.read_query, self.write_key, self.write_value, self.write_gate]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_router_weights(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute router weights for memory selection.

        Args:
            input_tensor: Current input tensor (batch_size, input_size)
            hidden: Current hidden state (batch_size, hidden_size)

        Returns:
            Router weights (batch_size, num_memories)
        """
        # Concatenate input and hidden state
        router_input = torch.cat([input_tensor, hidden], dim=-1)

        # Pass through router network and apply softmax
        router_logits = self.router(router_input)
        router_weights = F.softmax(router_logits, dim=-1)

        return router_weights

    def _read_from_memory(self, input_tensor: torch.Tensor, hidden: torch.Tensor,
                         router_weights: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention mechanism.

        Args:
            input_tensor: Current input tensor (batch_size, input_size)
            hidden: Current hidden state (batch_size, hidden_size)
            router_weights: Router weights (batch_size, num_memories)
            memory_state: Current memory state (batch_size, num_memories, memory_size, memory_dim)

        Returns:
            Context vector from memory (batch_size, memory_dim)
        """
        batch_size = input_tensor.shape[0]

        # Generate query for reading
        query_input = torch.cat([input_tensor, hidden], dim=-1)
        query = self.read_query(query_input)  # (batch_size, memory_dim)

        # Compute attention scores for each memory
        context_vectors = []

        for i in range(self.num_memories):
            memory_i = memory_state[:, i, :, :]  # (batch_size, memory_size, memory_dim)

            # Compute attention scores between query and memory slots
            # query: (batch_size, memory_dim), memory_i: (batch_size, memory_size, memory_dim)
            scores = torch.matmul(query.unsqueeze(1), memory_i.transpose(-2, -1))  # (batch_size, 1, memory_size)
            scores = scores.squeeze(1)  # (batch_size, memory_size)

            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores / math.sqrt(self.memory_dim), dim=-1)

            # Compute weighted sum of memory slots
            context_i = torch.matmul(attention_weights.unsqueeze(1), memory_i)  # (batch_size, 1, memory_dim)
            context_i = context_i.squeeze(1)  # (batch_size, memory_dim)

            context_vectors.append(context_i)

        # Stack context vectors and weight by router
        context_stack = torch.stack(context_vectors, dim=1)  # (batch_size, num_memories, memory_dim)

        # Apply router weights
        router_weights_expanded = router_weights.unsqueeze(-1)  # (batch_size, num_memories, 1)
        weighted_context = context_stack * router_weights_expanded  # (batch_size, num_memories, memory_dim)

        # Sum across memories
        final_context = weighted_context.sum(dim=1)  # (batch_size, memory_dim)

        return final_context

    def _write_to_memory(self, hidden: torch.Tensor, router_weights: torch.Tensor,
                        memory_state: torch.Tensor) -> torch.Tensor:
        """
        Write to memory using differentiable operations (NTM-style).

        Args:
            hidden: Current hidden state (batch_size, hidden_size)
            router_weights: Router weights (batch_size, num_memories)
            memory_state: Current memory state (batch_size, num_memories, memory_size, memory_dim)

        Returns:
            Updated memory state (batch_size, num_memories, memory_size, memory_dim)
        """
        batch_size = hidden.shape[0]

        # Generate write key, value, and erase vector
        write_key = self.write_key(hidden)  # (batch_size, memory_dim)
        write_value = self.write_value(hidden)  # (batch_size, memory_dim)
        write_gate_logit = self.write_gate(hidden)  # (batch_size, 1)
        write_gate = torch.sigmoid(write_gate_logit)  # (batch_size, 1)

        # Create erase vector (what to remove from memory)
        erase_vector = torch.sigmoid(write_value)  # (batch_size, memory_dim)

        # Create add vector (what to add to memory)
        add_vector = torch.tanh(write_value)  # (batch_size, memory_dim)

        new_memory_state = memory_state.clone()

        # Update each memory based on router weights
        for i in range(self.num_memories):
            memory_i = memory_state[:, i, :, :]  # (batch_size, memory_size, memory_dim)
            router_weight_i = router_weights[:, i:i+1]  # (batch_size, 1)

            # Compute similarity between write key and memory slots
            similarities = torch.matmul(write_key.unsqueeze(1), memory_i.transpose(-2, -1))  # (batch_size, 1, memory_size)
            similarities = similarities.squeeze(1)  # (batch_size, memory_size)

            # Apply softmax to get write attention weights
            write_attention = F.softmax(similarities / math.sqrt(self.memory_dim), dim=-1)  # (batch_size, memory_size)

            # Scale write attention by router weight and write gate
            scaled_attention = write_attention * router_weight_i * write_gate  # (batch_size, memory_size)

            # Erase operation: M_t = M_{t-1} * (1 - w_t * e_t)
            # scaled_attention: (batch_size, memory_size), erase_vector: (batch_size, memory_dim)
            erase_term = scaled_attention.unsqueeze(-1) * erase_vector.unsqueeze(1)  # (batch_size, memory_size, memory_dim)
            memory_after_erase = memory_i * (1 - erase_term)

            # Add operation: M_t = M_after_erase + w_t * a_t
            add_term = scaled_attention.unsqueeze(-1) * add_vector.unsqueeze(1)  # (batch_size, memory_size, memory_dim)
            memory_after_add = memory_after_erase + add_term

            new_memory_state[:, i, :, :] = memory_after_add

        return new_memory_state

    def forward(self, input_tensor: torch.Tensor, hidden: Optional[torch.Tensor] = None,
                memory_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MoM-GRU Cell.

        This implements the core GRU computation augmented with memory operations:
        1. Initialize or use provided memory state
        2. Compute router weights for memory selection
        3. Read context from memory
        4. Compute GRU gates with input + context
        5. Update hidden state
        6. Write to memory (differentiably)

        Args:
            input_tensor: Input tensor (batch_size, input_size)
            hidden: Previous hidden state (batch_size, hidden_size). If None, initialized to zeros.
            memory_state: Current memory state (batch_size, num_memories, memory_size, memory_dim).
                         If None, initialized from initial_memories.

        Returns:
            new_hidden: Updated hidden state (batch_size, hidden_size)
            new_hidden: Same as new_hidden (for compatibility with GRU interface)
            new_memory_state: Updated memory state (batch_size, num_memories, memory_size, memory_dim)
        """
        batch_size = input_tensor.shape[0]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size,
                               device=input_tensor.device, dtype=input_tensor.dtype)

        # Initialize memory state if not provided
        if memory_state is None:
            # Expand initial memories for the batch
            memory_state = self.initial_memories.unsqueeze(0).expand(
                batch_size, self.num_memories, self.memory_size, self.memory_dim
            ).contiguous()

        # Step 1: Compute router weights for memory selection
        router_weights = self._compute_router_weights(input_tensor, hidden)

        # Step 2: Read context from memory
        memory_context = self._read_from_memory(input_tensor, hidden, router_weights, memory_state)

        # Step 3: Augment input with memory context for GRU computation
        # Following d2l.ai GRU implementation but with memory context
        augmented_input = torch.cat([input_tensor, memory_context], dim=-1)

        # Compute reset gate: R_t = σ(X_t W_xr + H_{t-1} W_hr + b_r)
        reset_gate = torch.matmul(augmented_input, self.W_xr) + torch.matmul(hidden, self.W_hr)
        if self.b_r is not None:
            reset_gate = reset_gate + self.b_r
        reset_gate = torch.sigmoid(reset_gate)

        # Compute update gate: Z_t = σ(X_t W_xz + H_{t-1} W_hz + b_z)
        update_gate = torch.matmul(augmented_input, self.W_xz) + torch.matmul(hidden, self.W_hz)
        if self.b_z is not None:
            update_gate = update_gate + self.b_z
        update_gate = torch.sigmoid(update_gate)

        # Compute candidate hidden state: H̃_t = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh + b_h)
        candidate_hidden = torch.matmul(augmented_input, self.W_xh) + torch.matmul(reset_gate * hidden, self.W_hh)
        if self.b_h is not None:
            candidate_hidden = candidate_hidden + self.b_h
        candidate_hidden = torch.tanh(candidate_hidden)

        # Step 4: Compute new hidden state: H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H̃_t
        new_hidden = update_gate * hidden + (1 - update_gate) * candidate_hidden

        # Step 5: Write to memory using differentiable operations
        new_memory_state = self._write_to_memory(new_hidden, router_weights, memory_state)

        return new_hidden, new_hidden, new_memory_state

    def get_initial_memory_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get the initial memory state for a batch.

        Args:
            batch_size: Size of the batch
            device: Device to place the memory state on

        Returns:
            Initial memory state tensor (batch_size, num_memories, memory_size, memory_dim)
        """
        if device is None:
            device = self.initial_memories.device

        return self.initial_memories.unsqueeze(0).expand(
            batch_size, self.num_memories, self.memory_size, self.memory_dim
        ).contiguous().to(device)

    def reset_memory_parameters(self) -> None:
        """Reset the initial memory parameters to random values."""
        with torch.no_grad():
            self.initial_memories.data.normal_(0, 0.1)
