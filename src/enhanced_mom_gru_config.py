"""
Configuration system for Enhanced Memory-augmented GRU Cell (Enhanced MoM-GRU).

This module provides comprehensive configuration options for all enhancement features
including memory persistence and router stability.
Each feature can be independently enabled/disabled for ablation studies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch




@dataclass
class RouterStabilityConfig:
    """Configuration for router stabilization mechanisms."""
    
    # Enable/disable router stability features
    enabled: bool = True
    
    # Gumbel-Softmax exploration
    use_gumbel_softmax: bool = True
    initial_temperature: float = 1.0  # Starting temperature for exploration
    final_temperature: float = 0.1   # Final temperature for exploitation
    temperature_decay: float = 0.999  # Exponential decay rate for temperature
    min_temperature: float = 0.05    # Minimum temperature threshold
    
    # Loss-Free Balancing (LFB)
    use_loss_free_balancing: bool = True
    lfb_learning_rate: float = 0.01  # Learning rate for bias updates
    target_load_balance: Optional[float] = None  # Target load per bank (auto-calculated as 1/num_banks)
    balance_momentum: float = 0.9    # Momentum for load balance tracking
    
    # Router monitoring
    entropy_target: float = 1.2  # Target entropy for router distribution
    entropy_weight: float = 0.1  # Weight for entropy regularization (if used)



@dataclass
class MonitoringConfig:
    """Configuration for monitoring and diagnostics."""
    
    # Enable/disable monitoring features
    enabled: bool = True
    
    # Metrics tracking
    track_router_entropy: bool = True
    track_load_balance: bool = True
    track_memory_utilization: bool = True
    track_bank_specialization: bool = True
    
    # Logging frequency
    log_frequency: int = 100  # Log metrics every N steps
    detailed_logging: bool = False  # Enable detailed per-slot logging
    
    # Memory usage monitoring
    track_memory_usage: bool = True
    memory_usage_threshold: float = 0.8  # Warn if memory usage exceeds this


@dataclass
class EnhancedMoMGRUConfig:
    """
    Comprehensive configuration for Enhanced MoM-GRU Cell.
    
    This configuration allows independent control of all enhancement features,
    enabling thorough ablation studies and gradual feature adoption.
    """
    
    # Sub-configurations for each feature set
    router_stability: RouterStabilityConfig = field(default_factory=RouterStabilityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    
    # Training mode settings
    training_phase: str = "awake"  # "awake" or "sleep"
    step_count: int = 0           # Current training step (for scheduling)
    
    # Backward compatibility
    legacy_mode: bool = False     # Disable all enhancements for exact compatibility
    
    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        if self.legacy_mode:
            # Disable all enhancements for backward compatibility
            self.router_stability.enabled = False
            self.monitoring.enabled = False
    
    def get_current_temperature(self) -> float:
        """Get current Gumbel-Softmax temperature based on training progress."""
        if not self.router_stability.enabled or not self.router_stability.use_gumbel_softmax:
            return 1.0
        
        # Exponential decay from initial to final temperature
        decay_factor = self.router_stability.temperature_decay ** self.step_count
        current_temp = (
            self.router_stability.initial_temperature * decay_factor +
            self.router_stability.final_temperature * (1 - decay_factor)
        )
        return max(current_temp, self.router_stability.min_temperature)
    
    def should_run_sleep_phase(self) -> bool:
        """Determine if a sleep phase should be triggered."""
        # Sleep phases are no longer supported (continual learning removed)
        return False
    
    def increment_step(self):
        """Increment the step counter for scheduling."""
        self.step_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'router_stability': self.router_stability.__dict__,
            'monitoring': self.monitoring.__dict__,
            'device': str(self.device) if self.device else None,
            'dtype': str(self.dtype) if self.dtype else None,
            'training_phase': self.training_phase,
            'step_count': self.step_count,
            'legacy_mode': self.legacy_mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedMoMGRUConfig':
        """Create configuration from dictionary."""
        # This would need more sophisticated deserialization for production use
        return cls(**config_dict)


# Predefined configurations for common use cases
def get_baseline_config() -> EnhancedMoMGRUConfig:
    """Get configuration that matches original MoMGRUCell behavior."""
    return EnhancedMoMGRUConfig(legacy_mode=True)


def get_conservative_config() -> EnhancedMoMGRUConfig:
    """Get configuration with only essential enhancements enabled."""
    config = EnhancedMoMGRUConfig()
    config.router_stability.enabled = True
    config.monitoring.enabled = True
    return config


def get_full_config() -> EnhancedMoMGRUConfig:
    """Get configuration with all enhancements enabled."""
    config = EnhancedMoMGRUConfig()
    # All features enabled by default
    return config


def get_ablation_configs() -> Dict[str, EnhancedMoMGRUConfig]:
    """Get a set of configurations for ablation studies."""
    configs = {}
    
    # Baseline (no enhancements)
    configs['baseline'] = get_baseline_config()
    
    # Individual features
    configs['router_only'] = EnhancedMoMGRUConfig(
        router_stability=RouterStabilityConfig(enabled=True),
        monitoring=MonitoringConfig(enabled=False)
    )

    configs['monitoring_only'] = EnhancedMoMGRUConfig(
        router_stability=RouterStabilityConfig(enabled=False),
        monitoring=MonitoringConfig(enabled=True)
    )

    # Combinations
    configs['router_monitoring'] = EnhancedMoMGRUConfig(
        router_stability=RouterStabilityConfig(enabled=True),
        monitoring=MonitoringConfig(enabled=True)
    )
    
    configs['full'] = get_full_config()
    
    return configs
