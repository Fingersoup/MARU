"""
Unit tests for the Mamba block implementation.

This module contains tests to verify the correct functionality of the MambaBlock
and MambaConfig classes.
"""

import unittest
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mamba_block import MambaBlock, MambaConfig, create_mamba_block, selective_scan_sequential


class TestMambaConfig(unittest.TestCase):
    """Test cases for the MambaConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MambaConfig()
        
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.d_state, 16)
        self.assertEqual(config.d_conv, 4)
        self.assertEqual(config.expand, 2)
        self.assertEqual(config.d_inner, 512)  # expand * d_model
        self.assertEqual(config.dt_rank, 16)  # d_model // 16
        self.assertFalse(config.bias)
        self.assertTrue(config.conv_bias)
        self.assertTrue(config.pscan)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MambaConfig(
            d_model=128,
            d_state=32,
            d_conv=3,
            expand=4,
            dt_rank=8,
            bias=True,
            conv_bias=False
        )
        
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.d_state, 32)
        self.assertEqual(config.d_conv, 3)
        self.assertEqual(config.expand, 4)
        self.assertEqual(config.d_inner, 512)  # 4 * 128
        self.assertEqual(config.dt_rank, 8)
        self.assertTrue(config.bias)
        self.assertFalse(config.conv_bias)
    
    def test_dt_rank_auto_calculation(self):
        """Test automatic dt_rank calculation."""
        config = MambaConfig(d_model=64)
        self.assertEqual(config.dt_rank, 4)  # max(1, 64 // 16)
        
        config = MambaConfig(d_model=8)
        self.assertEqual(config.dt_rank, 1)  # max(1, 8 // 16)


class TestSelectiveScan(unittest.TestCase):
    """Test cases for the selective scan operation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 8
        self.d_inner = 4
        self.d_state = 3
        
        # Create test inputs
        self.u = torch.randn(self.batch_size, self.seq_len, self.d_inner)
        self.delta = torch.rand(self.batch_size, self.seq_len, self.d_inner) * 0.1
        self.A = -torch.rand(self.d_inner, self.d_state)
        self.B = torch.randn(self.batch_size, self.seq_len, self.d_state)
        self.C = torch.randn(self.batch_size, self.seq_len, self.d_state)
        self.D = torch.randn(self.d_inner)
    
    def test_selective_scan_shape(self):
        """Test that selective scan produces correct output shape."""
        y = selective_scan_sequential(self.u, self.delta, self.A, self.B, self.C, self.D)
        
        expected_shape = (self.batch_size, self.seq_len, self.d_inner)
        self.assertEqual(y.shape, expected_shape)
    
    def test_selective_scan_without_feedthrough(self):
        """Test selective scan without feedthrough connection."""
        y = selective_scan_sequential(self.u, self.delta, self.A, self.B, self.C, D=None)
        
        expected_shape = (self.batch_size, self.seq_len, self.d_inner)
        self.assertEqual(y.shape, expected_shape)
    
    def test_selective_scan_finite_output(self):
        """Test that selective scan produces finite outputs."""
        y = selective_scan_sequential(self.u, self.delta, self.A, self.B, self.C, self.D)
        
        self.assertTrue(torch.isfinite(y).all())


class TestMambaBlock(unittest.TestCase):
    """Test cases for the MambaBlock class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.d_model = 64
        self.batch_size = 2
        self.seq_len = 16
        
        self.config = MambaConfig(
            d_model=self.d_model,
            d_state=8,
            d_conv=4,
            expand=2
        )
        
        self.mamba = MambaBlock(self.config)
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model)
    
    def test_mamba_initialization(self):
        """Test that MambaBlock initializes correctly."""
        self.assertEqual(self.mamba.d_model, self.d_model)
        self.assertEqual(self.mamba.d_state, 8)
        self.assertEqual(self.mamba.d_inner, 128)  # expand * d_model
        
        # Check that all required layers are present
        self.assertIsInstance(self.mamba.in_proj, torch.nn.Linear)
        self.assertIsInstance(self.mamba.conv1d, torch.nn.Conv1d)
        self.assertIsInstance(self.mamba.x_proj, torch.nn.Linear)
        self.assertIsInstance(self.mamba.dt_proj, torch.nn.Linear)
        self.assertIsInstance(self.mamba.out_proj, torch.nn.Linear)
        
        # Check parameter shapes
        self.assertEqual(self.mamba.A_log.shape, (128, 8))  # (d_inner, d_state)
        self.assertEqual(self.mamba.D.shape, (128,))  # (d_inner,)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        output = self.mamba(self.input_tensor)
        
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
    
    def test_different_batch_sizes(self):
        """Test that the model works with different batch sizes."""
        for batch_size in [1, 4, 8]:
            input_tensor = torch.randn(batch_size, self.seq_len, self.d_model)
            output = self.mamba(input_tensor)
            
            expected_shape = (batch_size, self.seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
    
    def test_different_sequence_lengths(self):
        """Test that the model works with different sequence lengths."""
        for seq_len in [1, 8, 32, 64]:
            input_tensor = torch.randn(self.batch_size, seq_len, self.d_model)
            output = self.mamba(input_tensor)
            
            expected_shape = (self.batch_size, seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
    
    def test_output_finite(self):
        """Test that output values are finite."""
        output = self.mamba(self.input_tensor)
        
        self.assertTrue(torch.isfinite(output).all())
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        self.mamba.train()
        output = self.mamba(self.input_tensor)
        
        # Create dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for parameters
        for name, param in self.mamba.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}")
    
    def test_training_eval_modes(self):
        """Test switching between training and evaluation modes."""
        # Test training mode
        self.mamba.train()
        self.assertTrue(self.mamba.training)
        
        # Test evaluation mode
        self.mamba.eval()
        self.assertFalse(self.mamba.training)
    
    def test_parameters_require_grad(self):
        """Test that model parameters require gradients."""
        for param in self.mamba.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_create_mamba_block_factory(self):
        """Test the factory function for creating MambaBlock."""
        mamba = create_mamba_block(
            d_model=self.d_model,
            d_state=16,
            expand=4
        )
        
        self.assertIsInstance(mamba, MambaBlock)
        self.assertEqual(mamba.d_model, self.d_model)
        self.assertEqual(mamba.d_state, 16)
        self.assertEqual(mamba.d_inner, 256)  # 4 * 64


class TestMambaBlockEdgeCases(unittest.TestCase):
    """Test edge cases for the MambaBlock."""
    
    def test_small_model(self):
        """Test with very small model dimensions."""
        config = MambaConfig(d_model=8, d_state=4, expand=2)
        mamba = MambaBlock(config)
        
        input_tensor = torch.randn(1, 4, 8)
        output = mamba(input_tensor)
        
        self.assertEqual(output.shape, (1, 4, 8))
    
    def test_single_timestep(self):
        """Test with sequence length of 1."""
        config = MambaConfig(d_model=32, d_state=8)
        mamba = MambaBlock(config)
        
        input_tensor = torch.randn(2, 1, 32)
        output = mamba(input_tensor)
        
        self.assertEqual(output.shape, (2, 1, 32))
    
    def test_large_state_dimension(self):
        """Test with larger state dimension."""
        config = MambaConfig(d_model=64, d_state=64, expand=2)
        mamba = MambaBlock(config)
        
        input_tensor = torch.randn(2, 8, 64)
        output = mamba(input_tensor)
        
        self.assertEqual(output.shape, (2, 8, 64))


if __name__ == '__main__':
    unittest.main()
