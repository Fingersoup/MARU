"""
Core computational methods for Enhanced MoM-GRU Cell.

These methods implement the main forward pass logic with all enhancements.
They are designed to be added to the EnhancedMoMGRUCell class.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def compute_router_weights(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """
    Compute router weights for memory selection with stabilization.

    Args:
        input_tensor: Current input tensor (batch_size, input_size)
        hidden: Current hidden state (batch_size, hidden_size)

    Returns:
        Router weights (batch_size, num_memories)
    """
    # Concatenate input and hidden state
    router_input = torch.cat([input_tensor, hidden], dim=-1)

    # Pass through router network
    router_logits = self.router(router_input)
    
    # Apply Loss-Free Balancing
    router_logits = self.router_stabilizer.apply_loss_free_balancing(router_logits)
    
    # Apply Gumbel-Softmax with current temperature
    current_temperature = self.config.get_current_temperature()
    router_weights = self.router_stabilizer.apply_gumbel_softmax(router_logits, current_temperature)
    
    # Update load balancing statistics
    self.router_stabilizer.update_load_balance(router_weights)
    
    return router_weights


def read_from_memory(self, input_tensor: torch.Tensor, hidden: torch.Tensor,
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


def write_to_memory(self, hidden: torch.Tensor, router_weights: torch.Tensor,
                   memory_state: torch.Tensor) -> torch.Tensor:
    """
    Write to memory using differentiable operations with importance-based protection.

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

    # Get protection gates from importance tracker
    protection_gates = self.importance_tracker.compute_protection_gates(batch_size)

    new_memory_state = memory_state.clone()

    # Update each memory based on router weights
    for i in range(self.num_memories):
        memory_i = memory_state[:, i, :, :]  # (batch_size, memory_size, memory_dim)
        router_weight_i = router_weights[:, i:i+1]  # (batch_size, 1)
        protection_gate_i = protection_gates[:, i, :, :]  # (batch_size, memory_size, memory_dim)

        # Compute similarity between write key and memory slots
        similarities = torch.matmul(write_key.unsqueeze(1), memory_i.transpose(-2, -1))  # (batch_size, 1, memory_size)
        similarities = similarities.squeeze(1)  # (batch_size, memory_size)

        # Apply softmax to get write attention weights
        write_attention = F.softmax(similarities / math.sqrt(self.memory_dim), dim=-1)  # (batch_size, memory_size)

        # Scale write attention by router weight and write gate
        scaled_attention = write_attention * router_weight_i * write_gate  # (batch_size, memory_size)

        # Apply protection gates to erase vector (key enhancement!)
        protected_erase = erase_vector.unsqueeze(1) * protection_gate_i  # (batch_size, memory_size, memory_dim)

        # Erase operation with protection: M_t = M_{t-1} * (1 - w_t * e_t_protected)
        erase_term = scaled_attention.unsqueeze(-1) * protected_erase  # (batch_size, memory_size, memory_dim)
        memory_after_erase = memory_i * (1 - erase_term)

        # Add operation: M_t = M_after_erase + w_t * a_t (unchanged for plasticity)
        add_term = scaled_attention.unsqueeze(-1) * add_vector.unsqueeze(1)  # (batch_size, memory_size, memory_dim)
        memory_after_add = memory_after_erase + add_term

        new_memory_state[:, i, :, :] = memory_after_add

    return new_memory_state


def enhanced_forward(self, input_tensor: torch.Tensor, hidden: Optional[torch.Tensor] = None,
                    memory_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enhanced forward pass of the MoM-GRU Cell.

    This implements the core GRU computation augmented with enhanced memory operations:
    1. Initialize or use provided memory state
    2. Compute router weights with stabilization
    3. Read context from memory
    4. Compute GRU gates with input + context
    5. Update hidden state
    6. Write to memory with importance-based protection
    7. Update importance scores and collect monitoring statistics

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

    # Store current memory state for importance tracking
    self._current_memory_state = memory_state.detach()

    # Step 1: Compute router weights for memory selection (with stabilization)
    router_weights = self.compute_router_weights(input_tensor, hidden)

    # Step 2: Read context from memory
    memory_context = self.read_from_memory(input_tensor, hidden, router_weights, memory_state)

    # Step 3: Augment input with memory context for GRU computation
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

    # Step 5: Write to memory using enhanced operations with protection
    new_memory_state = self.write_to_memory(new_hidden, router_weights, memory_state)

    # Step 6: Collect monitoring statistics
    if self.config.monitoring.enabled and self.training:
        self._update_monitoring_stats(router_weights, memory_state, new_memory_state)

    # Step 7: Increment step counter for scheduling
    if self.training:
        self.config.increment_step()

    return new_hidden, new_hidden, new_memory_state
