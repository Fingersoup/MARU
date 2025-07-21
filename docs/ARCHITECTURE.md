# MARU Architecture Deep Dive

## Overview

MARU (Memory-Augmented Recurrent Unit) is a novel neural architecture that combines the parallel processing capabilities of state-space models with the persistent memory advantages of recurrent networks. This document provides a comprehensive technical overview of the architecture.

## Core Innovation

The key innovation of MARU is its **two-stage processing approach**:

1. **Parallel Stage**: Mamba block processes entire sequences in parallel
2. **Sequential Stage**: Memory-augmented GRU processes outputs recurrently

This design achieves **O(1) inference complexity** for autoregressive generation while maintaining rich contextual understanding through persistent memory mechanisms.

## Architecture Components

### 1. Mamba Block

The Mamba block implements a selective state-space model (S6) that can selectively focus on or ignore information based on input content.

**Key Features:**
- **Selective Scan**: Data-dependent state transitions
- **Parallel Processing**: Entire sequences processed simultaneously
- **Long-range Dependencies**: Captures relationships across long sequences

**Mathematical Foundation:**
```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t + D * x_t
```

Where A, B, C are learned parameters that depend on the input sequence.

**Implementation Details:**
- **d_model**: 640 (model dimension)
- **d_state**: 16 (state dimension)
- **d_conv**: 4 (convolution dimension)
- **expand**: 2 (expansion factor)

### 2. Memory-augmented GRU (MoM-GRU)

The MoM-GRU extends standard GRU cells with multiple persistent memory matrices.

**Components:**
- **Standard GRU Gates**: Reset and update gates for temporal dynamics
- **Memory Matrices**: Multiple persistent storage banks
- **Router Network**: Selects appropriate memory banks
- **Read/Write Heads**: Attention-based memory operations

**Memory Architecture:**
- **Number of Memory Banks**: 4
- **Memory Size per Bank**: 128 slots
- **Memory Dimension**: 64
- **Router Hidden Dimension**: 32

**Memory Operations:**
1. **Router Selection**: Determines which memory banks to access
2. **Read Operation**: Attention-based retrieval from selected banks
3. **Write Operation**: Updates memory content based on current state
4. **Memory Integration**: Combines read information with GRU processing

### 3. Enhanced MoM-GRU (Production Version)

The enhanced version includes additional stabilization and monitoring features:

**Enhancements:**
- **Router Stabilization**: Gumbel-Softmax for stable memory selection
- **Loss-Free Balancing**: Prevents memory bank collapse
- **Comprehensive Monitoring**: Tracks memory usage and router behavior
- **Attention-based Addressing**: Improved memory access patterns

## Processing Flow

### Forward Pass

1. **Input Embedding**
   ```python
   embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
   embedded = self.embed_proj(embedded)  # (batch, seq_len, d_model)
   ```

2. **Mamba Processing**
   ```python
   mamba_output = self.mamba_block(embedded)  # (batch, seq_len, d_model)
   ```

3. **GRU Input Projection**
   ```python
   gru_input = self.mamba_to_gru_proj(mamba_output)  # (batch, seq_len, hidden_size)
   ```

4. **Sequential MoM-GRU Processing**
   ```python
   for t in range(seq_len):
       hidden, memory_state = self.mom_gru_cell(
           gru_input[:, t], hidden, memory_state
       )
   ```

5. **Output Generation**
   ```python
   output = self.output_head(hidden)  # (batch, output_dim)
   ```

### Inference (O(1) Complexity)

For autoregressive generation, MARU processes one token at a time:

1. **Single Token Processing**: Only the new token goes through Mamba
2. **State Persistence**: Hidden and memory states are maintained
3. **Constant Time**: Each step takes O(1) time regardless of sequence length

## Memory Mechanisms

### Memory Bank Structure

Each memory bank consists of:
- **Key Matrix**: K ∈ ℝ^(memory_size × memory_dim)
- **Value Matrix**: V ∈ ℝ^(memory_size × memory_dim)
- **Usage Vector**: Tracks memory slot utilization

### Router Network

The router network determines memory bank selection:

```python
router_logits = self.router(hidden_state)  # (batch, num_memories)
router_weights = F.softmax(router_logits, dim=-1)
```

### Read Operation

Memory reading uses attention mechanisms:

```python
# Compute attention scores
attention_scores = torch.matmul(query, memory_keys.transpose(-2, -1))
attention_weights = F.softmax(attention_scores / sqrt(memory_dim), dim=-1)

# Read from memory
read_vector = torch.matmul(attention_weights, memory_values)
```

### Write Operation

Memory writing updates both keys and values:

```python
# Update memory based on current state
new_key = self.key_projection(hidden_state)
new_value = self.value_projection(hidden_state)

# Write to selected memory slots
memory_keys[selected_slots] = new_key
memory_values[selected_slots] = new_value
```

## Training Considerations

### Memory Initialization

- **Keys**: Random initialization with unit norm
- **Values**: Zero initialization
- **Usage**: Uniform initialization

### Gradient Flow

- **Mamba Block**: Standard backpropagation through parallel operations
- **MoM-GRU**: Backpropagation through time (BPTT) with memory gradients
- **Memory Operations**: Differentiable attention mechanisms

### Stability Features

- **Gradient Clipping**: Prevents exploding gradients in recurrent components
- **Router Stabilization**: Gumbel-Softmax prevents mode collapse
- **Memory Regularization**: Encourages diverse memory usage

## Performance Characteristics

### Computational Complexity

- **Training**: O(L) where L is sequence length
- **Inference**: O(1) per generated token
- **Memory**: O(M) where M is total memory capacity

### Memory Requirements

- **Model Parameters**: ~6.7M parameters
- **Memory Banks**: 4 × 128 × 64 = 32,768 memory parameters
- **Activation Memory**: Scales with batch size and sequence length

## Comparison with Other Architectures

| Architecture | Training Complexity | Inference Complexity | Memory Persistence |
|--------------|-------------------|---------------------|-------------------|
| Transformer | O(L²) | O(L) | Attention-based |
| Mamba | O(L) | O(1) | State-based |
| RNN/LSTM | O(L) | O(1) | Hidden state |
| **MARU** | **O(L)** | **O(1)** | **Persistent + State** |

## Future Directions

### Potential Improvements

1. **Hierarchical Memory**: Multi-level memory organization
2. **Adaptive Memory**: Dynamic memory allocation
3. **Cross-attention**: Memory sharing between sequences
4. **Compression**: Memory compression techniques

### Research Questions

1. How does memory capacity affect performance?
2. Can memory banks specialize for different types of information?
3. How does the architecture scale to very long sequences?
4. Can memory be shared across different tasks?

## Implementation Notes

### Key Design Decisions

1. **Character-level Tokenization**: Enables fine-grained control
2. **Conservative Memory Configuration**: Balances capability and stability
3. **Modular Design**: Easy to experiment with different components
4. **Production Features**: Router stabilization and monitoring

### Known Limitations

1. **Memory Capacity**: Fixed memory size may limit very long contexts
2. **Router Complexity**: Memory selection adds computational overhead
3. **Training Stability**: Requires careful hyperparameter tuning
4. **Interpretability**: Memory usage patterns need better visualization

---

This architecture represents a novel approach to combining the strengths of different neural network paradigms, achieving efficient inference while maintaining rich memory capabilities.
