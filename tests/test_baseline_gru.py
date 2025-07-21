"""
Unit tests for the BaselineGRU model.

This module contains tests to verify the correct functionality of the BaselineGRU
model implementation.
"""

import unittest
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baseline_gru import BaselineGRU, create_baseline_gru


class TestBaselineGRU(unittest.TestCase):
    """Test cases for the BaselineGRU model."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vocab_size = 100
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.batch_size = 4
        self.seq_len = 50
        
        self.model = BaselineGRU(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Create dummy input
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.embedding_dim, self.embedding_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.dropout, self.dropout)
        
        # Check that layers are properly initialized
        self.assertIsInstance(self.model.embedding, torch.nn.Embedding)
        self.assertIsInstance(self.model.gru, torch.nn.GRU)
        self.assertIsInstance(self.model.output_head, torch.nn.Sequential)
    
    def test_forward_pass_shape(self):
        """Test that the forward pass produces outputs with expected shapes."""
        output, hidden = self.model(self.input_ids)
        
        # Check output shape
        expected_output_shape = (self.batch_size, 1)  # output_dim = 1 by default
        self.assertEqual(output.shape, expected_output_shape)
        
        # Check hidden state shape
        expected_hidden_shape = (self.num_layers, self.batch_size, self.hidden_dim)
        self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_forward_pass_with_initial_hidden(self):
        """Test forward pass with provided initial hidden state."""
        device = self.input_ids.device
        initial_hidden = self.model.init_hidden(self.batch_size, device)
        
        output, hidden = self.model(self.input_ids, initial_hidden)
        
        # Check shapes
        expected_output_shape = (self.batch_size, 1)
        expected_hidden_shape = (self.num_layers, self.batch_size, self.hidden_dim)
        
        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_init_hidden(self):
        """Test the init_hidden method."""
        device = torch.device('cpu')
        hidden = self.model.init_hidden(self.batch_size, device)
        
        expected_shape = (self.num_layers, self.batch_size, self.hidden_dim)
        self.assertEqual(hidden.shape, expected_shape)
        self.assertEqual(hidden.device, device)
        
        # Check that hidden state is initialized to zeros
        self.assertTrue(torch.allclose(hidden, torch.zeros_like(hidden)))
    
    def test_different_batch_sizes(self):
        """Test that the model works with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            input_ids = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
            output, hidden = self.model(input_ids)
            
            expected_output_shape = (batch_size, 1)
            expected_hidden_shape = (self.num_layers, batch_size, self.hidden_dim)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_different_sequence_lengths(self):
        """Test that the model works with different sequence lengths."""
        for seq_len in [10, 25, 100, 200]:
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, seq_len))
            output, hidden = self.model(input_ids)
            
            expected_output_shape = (self.batch_size, 1)
            expected_hidden_shape = (self.num_layers, self.batch_size, self.hidden_dim)
            
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(hidden.shape, expected_hidden_shape)
    
    def test_create_baseline_gru_factory(self):
        """Test the factory function for creating BaselineGRU."""
        model = create_baseline_gru(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.assertIsInstance(model, BaselineGRU)
        self.assertEqual(model.vocab_size, self.vocab_size)
        self.assertEqual(model.embedding_dim, self.embedding_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
    
    def test_model_parameters_require_grad(self):
        """Test that model parameters require gradients."""
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_model_training_mode(self):
        """Test switching between training and evaluation modes."""
        # Test training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Test evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
    
    def test_output_range(self):
        """Test that output values are reasonable for position prediction."""
        output, _ = self.model(self.input_ids)
        
        # Output should be finite
        self.assertTrue(torch.isfinite(output).all())
        
        # For position prediction, outputs should typically be non-negative
        # (though the model might output negative values initially)
        # We just check that they're reasonable numbers
        self.assertTrue(output.abs().max() < 1000)  # Reasonable range check
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        self.model.train()
        output, _ = self.model(self.input_ids)
        
        # Create a dummy loss
        target = torch.randint(0, self.seq_len, (self.batch_size, 1)).float()
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}")


class TestBaselineGRUEdgeCases(unittest.TestCase):
    """Test edge cases for the BaselineGRU model."""
    
    def test_single_layer_gru(self):
        """Test model with single GRU layer."""
        model = BaselineGRU(vocab_size=50, num_layers=1)
        input_ids = torch.randint(0, 50, (2, 10))
        
        output, hidden = model(input_ids)
        
        self.assertEqual(output.shape, (2, 1))
        self.assertEqual(hidden.shape, (1, 2, 256))  # default hidden_dim
    
    def test_custom_output_dim(self):
        """Test model with custom output dimension."""
        output_dim = 5
        model = BaselineGRU(vocab_size=50, output_dim=output_dim)
        input_ids = torch.randint(0, 50, (2, 10))
        
        output, hidden = model(input_ids)
        
        self.assertEqual(output.shape, (2, output_dim))


if __name__ == '__main__':
    unittest.main()
