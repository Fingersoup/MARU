"""
Unit tests for the Memory-augmented GRU Cell (MoM-GRU).

This module tests the MoMGRUCell implementation to ensure it:
1. Produces outputs of the correct shape
2. Maintains memory state across timesteps
3. Handles different batch sizes and input dimensions
4. Integrates properly with the MARU architecture
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mom_gru_cell import MoMGRUCell


class TestMoMGRUCell(unittest.TestCase):
    """Test basic functionality of the MoM-GRU Cell."""
    
    def setUp(self):
        """Set up test parameters."""
        self.input_size = 32
        self.hidden_size = 64
        self.memory_size = 16
        self.memory_dim = 32
        self.num_memories = 4
        self.batch_size = 2
        
        self.cell = MoMGRUCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            memory_dim=self.memory_dim,
            num_memories=self.num_memories
        )
    
    def test_initialization(self):
        """Test that the cell initializes with correct parameters."""
        self.assertEqual(self.cell.input_size, self.input_size)
        self.assertEqual(self.cell.hidden_size, self.hidden_size)
        self.assertEqual(self.cell.memory_size, self.memory_size)
        self.assertEqual(self.cell.memory_dim, self.memory_dim)
        self.assertEqual(self.cell.num_memories, self.num_memories)

        # Check initial memory shape
        expected_memory_shape = (self.num_memories, self.memory_size, self.memory_dim)
        self.assertEqual(self.cell.initial_memories.shape, expected_memory_shape)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_tensor = torch.randn(self.batch_size, self.input_size)
        hidden = torch.randn(self.batch_size, self.hidden_size)

        new_hidden, output, memory_state = self.cell(input_tensor, hidden)

        # Check output shapes
        expected_shape = (self.batch_size, self.hidden_size)
        self.assertEqual(new_hidden.shape, expected_shape)
        self.assertEqual(output.shape, expected_shape)

        # Check memory state shape
        expected_memory_shape = (self.batch_size, self.cell.num_memories, self.cell.memory_size, self.cell.memory_dim)
        self.assertEqual(memory_state.shape, expected_memory_shape)
        
        # Check that outputs are the same (GRU interface compatibility)
        self.assertTrue(torch.equal(new_hidden, output))
    
    def test_forward_pass_no_initial_hidden(self):
        """Test forward pass without providing initial hidden state."""
        input_tensor = torch.randn(self.batch_size, self.input_size)

        new_hidden, output, memory_state = self.cell(input_tensor)

        expected_shape = (self.batch_size, self.hidden_size)
        self.assertEqual(new_hidden.shape, expected_shape)
        self.assertEqual(output.shape, expected_shape)

        # Check memory state shape
        expected_memory_shape = (self.batch_size, self.cell.num_memories, self.cell.memory_size, self.cell.memory_dim)
        self.assertEqual(memory_state.shape, expected_memory_shape)
    
    def test_single_timestep_processing(self):
        """Test that the cell processes one timestep at a time correctly."""
        input_tensor = torch.randn(self.batch_size, self.input_size)
        hidden = torch.zeros(self.batch_size, self.hidden_size)

        # Process one timestep
        hidden1, _, memory_state1 = self.cell(input_tensor, hidden)

        # Process another timestep with the updated hidden state and memory
        hidden2, _, memory_state2 = self.cell(input_tensor, hidden1, memory_state1)

        # Hidden states should be different (unless by coincidence)
        self.assertFalse(torch.equal(hidden1, hidden2))

        # Memory states should also be different
        self.assertFalse(torch.equal(memory_state1, memory_state2))
    
    def test_memory_operations(self):
        """Test memory read/write operations."""
        input_tensor = torch.randn(self.batch_size, self.input_size)

        # Get initial memory state
        initial_memory = self.cell.get_initial_memory_state(self.batch_size)

        # Process a timestep (should modify memory)
        _, _, updated_memory = self.cell(input_tensor, memory_state=initial_memory)

        # Memory should have changed (unless by coincidence)
        self.assertFalse(torch.equal(initial_memory, updated_memory))
    
    def test_memory_reset(self):
        """Test memory reset functionality."""
        input_tensor = torch.randn(self.batch_size, self.input_size)

        # Get initial memory state
        memory_state = self.cell.get_initial_memory_state(self.batch_size)

        # Process some timesteps to modify memory
        for _ in range(3):
            _, _, memory_state = self.cell(input_tensor, memory_state=memory_state)

        memory_before_reset = memory_state.clone()

        # Reset memory parameters
        self.cell.reset_memory_parameters()

        memory_after_reset = self.cell.get_initial_memory_state(self.batch_size)

        # Memory should be different after reset
        self.assertFalse(torch.equal(memory_before_reset, memory_after_reset))
    
    def test_memory_state_management(self):
        """Test getting initial memory state."""
        # Get initial memory state
        initial_state = self.cell.get_initial_memory_state(self.batch_size)

        # Check shape
        expected_shape = (self.batch_size, self.cell.num_memories, self.cell.memory_size, self.cell.memory_dim)
        self.assertEqual(initial_state.shape, expected_shape)

        # Test with different batch size
        different_batch_size = 5
        different_state = self.cell.get_initial_memory_state(different_batch_size)
        expected_different_shape = (different_batch_size, self.cell.num_memories, self.cell.memory_size, self.cell.memory_dim)
        self.assertEqual(different_state.shape, expected_different_shape)
    
    def test_different_batch_sizes(self):
        """Test that the cell works with different batch sizes."""
        for batch_size in [1, 3, 8, 16]:
            input_tensor = torch.randn(batch_size, self.input_size)
            hidden = torch.randn(batch_size, self.hidden_size)

            new_hidden, output, memory_state = self.cell(input_tensor, hidden)

            expected_shape = (batch_size, self.hidden_size)
            self.assertEqual(new_hidden.shape, expected_shape)
            self.assertEqual(output.shape, expected_shape)

            expected_memory_shape = (batch_size, self.cell.num_memories, self.cell.memory_size, self.cell.memory_dim)
            self.assertEqual(memory_state.shape, expected_memory_shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the cell correctly."""
        input_tensor = torch.randn(self.batch_size, self.input_size, requires_grad=True)
        hidden = torch.randn(self.batch_size, self.hidden_size, requires_grad=True)

        new_hidden, _, memory_state = self.cell(input_tensor, hidden)

        # Compute a simple loss
        loss = new_hidden.sum() + memory_state.sum()  # Include memory in loss to ensure all paths get gradients
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(hidden.grad)

        # Check that key cell parameters have gradients
        # Focus on parameters that should definitely have gradients
        key_params = [
            self.cell.W_xr, self.cell.W_hr, self.cell.W_xz, self.cell.W_hz,
            self.cell.W_xh, self.cell.W_hh, self.cell.initial_memories
        ]

        for param in key_params:
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {param} should have gradients")


class TestMoMGRUCellEdgeCases(unittest.TestCase):
    """Test edge cases for the MoM-GRU Cell."""
    
    def test_minimal_configuration(self):
        """Test with minimal memory configuration."""
        cell = MoMGRUCell(
            input_size=4,
            hidden_size=8,
            memory_size=2,
            memory_dim=4,
            num_memories=2
        )

        input_tensor = torch.randn(1, 4)
        new_hidden, output, memory_state = cell(input_tensor)

        self.assertEqual(new_hidden.shape, (1, 8))
        self.assertEqual(output.shape, (1, 8))
        expected_memory_shape = (1, 2, 2, 4)  # (batch, num_memories, memory_size, memory_dim)
        self.assertEqual(memory_state.shape, expected_memory_shape)
    
    def test_large_configuration(self):
        """Test with larger memory configuration."""
        cell = MoMGRUCell(
            input_size=128,
            hidden_size=256,
            memory_size=64,
            memory_dim=128,
            num_memories=8
        )

        input_tensor = torch.randn(4, 128)
        new_hidden, output, memory_state = cell(input_tensor)

        self.assertEqual(new_hidden.shape, (4, 256))
        self.assertEqual(output.shape, (4, 256))
        expected_memory_shape = (4, 8, 64, 128)  # (batch, num_memories, memory_size, memory_dim)
        self.assertEqual(memory_state.shape, expected_memory_shape)
    
    def test_no_bias(self):
        """Test cell without bias terms."""
        cell = MoMGRUCell(
            input_size=16,
            hidden_size=32,
            bias=False
        )

        input_tensor = torch.randn(2, 16)
        new_hidden, output, memory_state = cell(input_tensor)

        self.assertEqual(new_hidden.shape, (2, 32))
        self.assertEqual(output.shape, (2, 32))

        # Check that bias parameters are None
        self.assertIsNone(cell.b_r)
        self.assertIsNone(cell.b_z)
        self.assertIsNone(cell.b_h)


class TestMoMGRUCellIntegration(unittest.TestCase):
    """Test integration with other MARU components."""
    
    def test_sequence_processing(self):
        """Test processing a sequence of inputs."""
        cell = MoMGRUCell(input_size=16, hidden_size=32)

        batch_size = 2
        seq_len = 5

        # Create a sequence of inputs
        inputs = [torch.randn(batch_size, 16) for _ in range(seq_len)]

        hidden = None
        memory_state = None
        outputs = []

        # Process sequence step by step
        for input_step in inputs:
            hidden, output, memory_state = cell(input_step, hidden, memory_state)
            outputs.append(output)

        # Check that we got the right number of outputs
        self.assertEqual(len(outputs), seq_len)

        # Check output shapes
        for output in outputs:
            self.assertEqual(output.shape, (batch_size, 32))
    
    def test_compatibility_with_embedding(self):
        """Test compatibility with embedding layers."""
        vocab_size = 100
        embedding_dim = 32
        hidden_size = 64

        embedding = nn.Embedding(vocab_size, embedding_dim)
        cell = MoMGRUCell(input_size=embedding_dim, hidden_size=hidden_size)

        # Create token IDs
        token_ids = torch.randint(0, vocab_size, (2, 1))  # Single timestep

        # Embed tokens
        embedded = embedding(token_ids).squeeze(1)  # Remove sequence dimension

        # Process through cell
        new_hidden, output, memory_state = cell(embedded)

        self.assertEqual(new_hidden.shape, (2, hidden_size))
        self.assertEqual(output.shape, (2, hidden_size))


if __name__ == '__main__':
    unittest.main()
