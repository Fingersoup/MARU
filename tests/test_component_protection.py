#!/usr/bin/env python3
"""
Test Component-Aware Parameter Protection Implementation

This script tests the new component-aware parameter protection feature
that replaces uniform protection with component-specific values.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from maru import MARU, MARUConfig
from sleep_orchestrator import SleepOrchestrator
from consolidation_manager import ConsolidationManager
from importance_tracker import ImportanceTracker
from continual_learning_config import ContinualLearningConfig, ComponentProtectionConfig, SleepPhaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_component_aware_protection():
    """Test component-aware parameter protection configuration and functionality."""
    
    logger.info("=== Testing Component-Aware Parameter Protection ===")
    
    # Create MARU model
    config = MARUConfig(
        vocab_size=256,
        d_model=64,
        hidden_size=64,
        memory_size=32,
        memory_dim=32,
        num_memories=4
    )
    
    model = MARU(config)
    logger.info(f"✓ MARU model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test 1: Component-aware protection configuration
    logger.info("1. Testing component-aware protection configuration...")
    
    # Create component-aware protection config
    protection_config = ComponentProtectionConfig(
        strength_mamba=4.0,
        strength_router=4.0,
        strength_memory=0.5
    )
    
    sleep_config = SleepPhaseConfig(
        learning_rate=0.001,
        duration=2,
        protection=protection_config
    )
    
    continual_config = ContinualLearningConfig(sleep_phase=sleep_config)
    
    # Create components
    consolidation_config = {
        'vae_buffer_size': 1000,
        'core_set_size': 64,
        'sampling_rate': 0.1,
        'confidence_bias': 2.0
    }
    consolidation_manager = ConsolidationManager(consolidation_config)
    importance_tracker = ImportanceTracker(model, ema_decay=0.999)
    
    # Create sleep orchestrator with component-aware protection
    sleep_orchestrator = SleepOrchestrator(
        model, consolidation_manager, importance_tracker, continual_config
    )
    
    # Verify component-aware protection is enabled
    assert sleep_orchestrator.use_component_aware_protection, "Component-aware protection should be enabled"
    assert sleep_orchestrator.protection_strengths['mamba'] == 4.0, "Mamba protection should be 4.0"
    assert sleep_orchestrator.protection_strengths['router'] == 4.0, "Router protection should be 4.0"
    assert sleep_orchestrator.protection_strengths['memory'] == 0.5, "Memory protection should be 0.5"
    
    logger.info("✓ Component-aware protection configuration works")
    
    # Test 2: Parameter name mapping
    logger.info("2. Testing parameter name to component mapping...")
    
    test_cases = [
        ("mamba_block.in_proj.weight", 4.0, "mamba"),
        ("mamba_to_gru_proj.weight", 4.0, "mamba"),
        ("embedding.weight", 4.0, "mamba"),
        ("mom_gru_cell.router.weight", 4.0, "router"),
        ("mom_gru_cell.initial_memories", 0.5, "memory"),
        ("mom_gru_cell.gru_cell.weight_ih", 0.5, "memory"),
        ("output_head.weight", 0.5, "memory")  # Unknown -> memory
    ]
    
    for param_name, expected_strength, expected_component in test_cases:
        actual_strength = sleep_orchestrator._get_component_protection_strength(param_name)
        assert actual_strength == expected_strength, f"Parameter '{param_name}' should have strength {expected_strength}, got {actual_strength}"
        logger.info(f"  ✓ {param_name} -> {expected_component} (strength: {actual_strength})")
    
    logger.info("✓ Parameter name mapping works correctly")
    
    # Test 3: Backward compatibility with uniform protection
    logger.info("3. Testing backward compatibility with uniform protection...")
    
    # Create config with old-style uniform protection
    uniform_config = {
        'sleep_phase': {
            'learning_rate': 0.001,
            'duration': 2,
            'protection_strength': 2.0,
            'triggers': {'memory_utilization': 0.8, 'router_entropy': 1.5}
        }
    }
    
    uniform_orchestrator = SleepOrchestrator(
        model, consolidation_manager, importance_tracker, uniform_config
    )
    
    # Verify uniform protection is used
    assert not uniform_orchestrator.use_component_aware_protection, "Should use uniform protection"
    assert uniform_orchestrator.protection_strength == 2.0, "Uniform protection strength should be 2.0"
    
    logger.info("✓ Backward compatibility with uniform protection works")
    
    # Test 4: Actual parameter protection during gradient update
    logger.info("4. Testing actual parameter protection during gradient update...")
    
    # Create some dummy gradients
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 0.1
    
    # Update importance scores with dummy values
    for name, param in model.named_parameters():
        importance_tracker.omega_scores[name] = torch.ones_like(param) * 0.5
    
    # Apply protected gradient update
    grad_stats = sleep_orchestrator._apply_protected_gradient_update()
    
    # Verify gradients were scaled
    assert grad_stats['scaled_params'] > 0, "Some parameters should have been scaled"
    logger.info(f"✓ Protected gradient update scaled {grad_stats['scaled_params']} parameters")
    
    logger.info("=== ALL COMPONENT-AWARE PROTECTION TESTS PASSED! ===")
    logger.info("✅ Component-aware parameter protection is working correctly")
    logger.info("✅ Mamba backbone: HIGH protection (4.0)")
    logger.info("✅ Router policy: HIGH protection (4.0)")
    logger.info("✅ Memory banks: LOW protection (0.5) for plasticity")
    logger.info("✅ Backward compatibility maintained")

if __name__ == "__main__":
    test_component_aware_protection()
