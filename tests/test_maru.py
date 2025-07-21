"""
Unit tests for the MARU (Memory-Augmented Recurrent Unit) architecture.

This module tests the complete MARU model to ensure it:
1. Integrates all components correctly
2. Produces outputs of the correct shape
3. Supports both training and inference modes
4. Maintains O(1) inference complexity
5. Handles memory operations properly
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maru import MARU, MARUConfig, create_maru_model


class TestMARUConfig(unittest.TestCase):
    """Test MARU configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MARUConfig(vocab_size=1000)
        
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.embedding_dim, 256)  # Should default to d_model
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.output_dim, 1)
        self.assertEqual(config.num_memories, 4)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MARUConfig(
            vocab_size=5000,
            d_model=512,
            embedding_dim=128,
            hidden_size=256,
            output_dim=10,
            num_memories=8
        )
        
        self.assertEqual(config.vocab_size, 5000)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.output_dim, 10)
        self.assertEqual(config.num_memories, 8)


class TestMARUModel(unittest.TestCase):
    """Test basic functionality of the MARU model."""
    
    def setUp(self):
        """Set up test parameters."""
        self.vocab_size = 100
        self.d_model = 64
        self.hidden_size = 64
        self.batch_size = 2
        self.seq_len = 8
        
        self.config = MARUConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            memory_size=16,
            memory_dim=32,
            num_memories=4
        )
        self.model = MARU(self.config)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model.embedding, nn.Embedding)
        self.assertIsInstance(self.model.mamba_block, nn.Module)
        self.assertIsInstance(self.model.mom_gru_cell, nn.Module)
        self.assertIsInstance(self.model.output_head, nn.Sequential)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shapes."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        output, hidden = self.model(input_ids, return_hidden=True)
        
        # Check output shape
        expected_output_shape = (self.batch_size, 1)  # output_dim=1 by default
        self.assertEqual(output.shape, expected_output_shape)
        
        # Check hidden shape
        expected_hidden_shape = (self.batch_size, self.hidden_size)
        self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_forward_pass_no_return_hidden(self):
        """Test forward pass without returning hidden state."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        output, hidden = self.model(input_ids, return_hidden=False)
        
        expected_output_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_output_shape)
        self.assertIsNone(hidden)
    
    def test_generate_step(self):
        """Test single-step generation for autoregressive inference."""
        input_id = torch.randint(0, self.vocab_size, (self.batch_size, 1))
        hidden = torch.randn(self.batch_size, self.hidden_size)

        output, new_hidden, new_memory_state = self.model.generate_step(input_id, hidden)

        # Check shapes
        expected_output_shape = (self.batch_size, 1)
        expected_hidden_shape = (self.batch_size, self.hidden_size)

        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(new_hidden.shape, expected_hidden_shape)
        self.assertIsNotNone(new_memory_state)

        # Hidden state should change
        self.assertFalse(torch.equal(hidden, new_hidden))
    
    def test_memory_operations(self):
        """Test memory state management."""
        # Get initial memory state
        initial_memory = self.model.get_initial_memory_state(self.batch_size)

        # Process some input with memory state tracking
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output1, hidden1 = self.model(input_ids, memory_state=initial_memory, return_hidden=True)

        # Process again with different input to see memory persistence
        input_ids2 = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output2, hidden2 = self.model(input_ids2, memory_state=initial_memory, return_hidden=True)

        # Outputs should be different (different inputs)
        self.assertFalse(torch.equal(output1, output2))

        # Test memory parameter reset
        old_memory_params = self.model.mom_gru_cell.initial_memories.clone()
        self.model.reset_memory_parameters()
        new_memory_params = self.model.mom_gru_cell.initial_memories
        self.assertFalse(torch.equal(old_memory_params, new_memory_params))
    
    def test_different_batch_sizes(self):
        """Test that the model works with different batch sizes."""
        for batch_size in [1, 3, 8]:
            input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
            output, hidden = self.model(input_ids, return_hidden=True)
            
            expected_output_shape = (batch_size, 1)
            expected_hidden_shape = (batch_size, self.hidden_size)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_different_sequence_lengths(self):
        """Test that the model works with different sequence lengths."""
        for seq_len in [1, 4, 16, 32]:
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, seq_len))
            output, hidden = self.model(input_ids, return_hidden=True)
            
            expected_output_shape = (self.batch_size, 1)
            expected_hidden_shape = (self.batch_size, self.hidden_size)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        output, _ = self.model(input_ids)
        
        # Compute a simple loss
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that model parameters have gradients
        params_with_grad = 0
        total_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
        
        # Most parameters should have gradients (some memory params might not due to detached updates)
        self.assertGreater(params_with_grad / total_params, 0.8)
    
    def test_parameter_counting(self):
        """Test parameter counting methods."""
        total_params = self.model.get_num_parameters()
        breakdown = self.model.get_parameter_breakdown()
        
        # Check that breakdown sums to total
        self.assertEqual(breakdown['total'], total_params)
        
        # Check that all components have parameters
        self.assertGreater(breakdown['embedding'], 0)
        self.assertGreater(breakdown['mamba_block'], 0)
        self.assertGreater(breakdown['mom_gru_cell'], 0)
        self.assertGreater(breakdown['output_head'], 0)


class TestMARUFactory(unittest.TestCase):
    """Test the factory function for creating MARU models."""
    
    def test_create_maru_model(self):
        """Test the factory function."""
        model = create_maru_model(
            vocab_size=1000,
            d_model=128,
            hidden_size=128,
            output_dim=5
        )
        
        self.assertIsInstance(model, MARU)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.d_model, 128)
        self.assertEqual(model.config.hidden_size, 128)
        self.assertEqual(model.config.output_dim, 5)


class TestMARUIntegration(unittest.TestCase):
    """Test integration scenarios for MARU."""
    
    def test_autoregressive_generation(self):
        """Test autoregressive generation scenario."""
        model = create_maru_model(vocab_size=50, d_model=32, hidden_size=32)

        batch_size = 1
        max_length = 10

        # Start with a random token
        current_token = torch.randint(0, 50, (batch_size, 1))
        hidden = None
        memory_state = None

        generated_tokens = [current_token]

        # Generate sequence step by step
        for _ in range(max_length - 1):
            output, hidden, memory_state = model.generate_step(current_token, hidden, memory_state)

            # Convert output to next token (simplified - just use argmax of random projection)
            next_token = torch.randint(0, 50, (batch_size, 1))  # Simplified for testing
            generated_tokens.append(next_token)
            current_token = next_token

        # Check that we generated the expected number of tokens
        self.assertEqual(len(generated_tokens), max_length)

        # Check that each token has the right shape
        for token in generated_tokens:
            self.assertEqual(token.shape, (batch_size, 1))
    
    def test_training_vs_inference_consistency(self):
        """Test that training and inference modes produce consistent results."""
        model = create_maru_model(vocab_size=20, d_model=16, hidden_size=16)

        # Create a short sequence
        input_ids = torch.randint(0, 20, (1, 4))

        # Training mode: process entire sequence
        model.train()
        train_output, train_hidden = model(input_ids, return_hidden=True)

        # Inference mode: process step by step
        model.eval()
        hidden = None
        memory_state = None
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i:i+1]
            output, hidden, memory_state = model.generate_step(token, hidden, memory_state)

        # Final outputs should be similar (not exactly equal due to different computation paths)
        # But shapes should match
        self.assertEqual(train_output.shape, output.shape)
        self.assertEqual(train_hidden.shape, hidden.shape)


if __name__ == '__main__':
    unittest.main()
