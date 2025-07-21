"""
Knowledge Distillation Configuration for MARU

This module provides configuration classes for knowledge distillation with teacher models.
It integrates with the existing MARU configuration system and supports various distillation
strategies and teacher model configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import yaml
from pathlib import Path


@dataclass
class TeacherModelConfig:
    """Configuration for teacher models used in knowledge distillation."""
    
    # Model identification
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal"  # "causal" or "masked"
    
    # Model loading parameters
    device: str = "auto"
    torch_dtype: str = "auto"  # "float16", "float32", "auto"
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    
    # Inference parameters
    temperature: float = 3.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    
    # Memory and performance
    max_memory_gb: float = 4.0
    batch_size: int = 8
    max_sequence_length: int = 512
    
    # Logit storage
    logit_storage_path: str = "data/teacher_logits/"
    logit_format: str = "compressed_numpy"  # "compressed_numpy", "torch", "hdf5"
    logit_dtype: str = "float16"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.model_type not in ["causal", "masked"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        if self.torch_dtype not in ["float16", "float32", "auto"]:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}")
        
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.max_sequence_length <= 0:
            raise ValueError("Max sequence length must be positive")


@dataclass
class DistillationLossConfig:
    """Configuration for distillation loss computation."""
    
    # Loss weighting
    alpha: float = 0.7  # Weight for distillation loss (vs hard targets)
    temperature: float = 3.0  # Temperature for KL divergence
    
    # Loss components
    use_hard_targets: bool = True
    use_soft_targets: bool = True
    use_attention_transfer: bool = False
    use_feature_matching: bool = False
    
    # Advanced loss options
    adaptive_alpha: bool = False  # Adapt alpha based on training progress
    alpha_schedule: str = "constant"  # "constant", "linear_decay", "cosine_decay"
    alpha_min: float = 0.1  # Minimum alpha value for scheduling

    # Adaptive temperature scaling
    adaptive_temperature: bool = False  # Enable adaptive temperature scaling
    temperature_method: str = "sharpness_based"  # "fixed", "curriculum", "sharpness_based", "logit_correlation", "combined"
    min_temperature: float = 1.0  # Minimum temperature value
    max_temperature: float = 8.0  # Maximum temperature value
    temperature_adaptation_rate: float = 0.1  # Rate of temperature adaptation

    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_schedule: str = "linear"  # "linear", "exponential", "step"
    curriculum_steps: int = 1000

    # Alternative loss functions
    logit_loss_type: str = "kl_divergence"  # "kl_divergence" or "mse"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")

        if not 0 <= self.alpha_min <= 1:
            raise ValueError("Alpha min must be between 0 and 1")

        if self.alpha_min > self.alpha:
            raise ValueError("Alpha min must be <= alpha")

        # Validate adaptive temperature parameters
        if self.min_temperature <= 0:
            raise ValueError("Min temperature must be positive")

        if self.max_temperature <= self.min_temperature:
            raise ValueError("Max temperature must be greater than min temperature")

        if self.temperature_adaptation_rate <= 0:
            raise ValueError("Temperature adaptation rate must be positive")

        valid_methods = ["fixed", "curriculum", "sharpness_based", "logit_correlation", "combined"]
        if self.temperature_method not in valid_methods:
            raise ValueError(f"Temperature method must be one of: {valid_methods}")


@dataclass
class VocabularyAlignmentConfig:
    """Configuration for vocabulary alignment between teacher and student."""
    
    # Alignment strategy
    strategy: str = "exact_match"  # "exact_match", "fuzzy_match", "embedding_based"
    
    # Fuzzy matching parameters
    similarity_threshold: float = 0.8
    use_subword_matching: bool = True
    
    # Embedding-based alignment
    embedding_model: Optional[str] = None
    embedding_cache_dir: Optional[str] = None
    
    # Fallback handling
    unk_token_strategy: str = "map_to_unk"  # "map_to_unk", "ignore", "random"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        valid_strategies = ["exact_match", "fuzzy_match", "embedding_based"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")


@dataclass
class DistillationConfig:
    """Main configuration class for knowledge distillation."""
    
    # Enable/disable distillation
    enabled: bool = False
    
    # Teacher model configuration
    teacher_model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    
    # Loss configuration
    loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    
    # Vocabulary alignment
    vocab_alignment: VocabularyAlignmentConfig = field(default_factory=VocabularyAlignmentConfig)
    
    # Training integration
    distill_every_n_steps: int = 1  # How often to apply distillation
    warmup_steps: int = 0  # Steps before starting distillation
    
    # Logit extraction and caching
    extract_logits_offline: bool = True  # Pre-extract logits vs online extraction
    logit_cache_size: int = 10000  # Number of cached logit tensors
    
    # Monitoring and logging
    log_distillation_metrics: bool = True
    log_teacher_student_agreement: bool = True
    save_teacher_outputs: bool = False
    
    # Performance optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        self.teacher_model.validate()
        self.loss.validate()
        self.vocab_alignment.validate()
        
        if self.distill_every_n_steps <= 0:
            raise ValueError("Distill every n steps must be positive")
        
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")
        
        if self.logit_cache_size <= 0:
            raise ValueError("Logit cache size must be positive")


# Predefined configurations for common use cases
def get_conservative_distillation_config() -> DistillationConfig:
    """Get a conservative distillation configuration for stable training."""
    return DistillationConfig(
        enabled=True,
        teacher_model=TeacherModelConfig(
            model_name="distilgpt2",
            temperature=3.0,
            max_memory_gb=2.0,
            batch_size=4
        ),
        loss=DistillationLossConfig(
            alpha=0.5,  # Conservative weighting
            temperature=3.0,
            adaptive_alpha=False
        ),
        distill_every_n_steps=2,  # Less frequent distillation
        warmup_steps=100
    )


def get_aggressive_distillation_config() -> DistillationConfig:
    """Get an aggressive distillation configuration for maximum knowledge transfer."""
    return DistillationConfig(
        enabled=True,
        teacher_model=TeacherModelConfig(
            model_name="microsoft/DialoGPT-medium",
            temperature=2.0,
            max_memory_gb=6.0,
            batch_size=8
        ),
        loss=DistillationLossConfig(
            alpha=0.8,  # High distillation weight
            temperature=2.0,
            adaptive_alpha=True,
            curriculum_learning=True
        ),
        distill_every_n_steps=1,  # Every step
        warmup_steps=50
    )


def get_character_level_distillation_config() -> DistillationConfig:
    """Get configuration for character-level teacher models."""
    return DistillationConfig(
        enabled=True,
        teacher_model=TeacherModelConfig(
            model_name="gpt2",
            temperature=4.0,  # Higher temperature for character-level
            max_memory_gb=3.0,
            batch_size=6,
            max_sequence_length=256  # Shorter sequences for char-level
        ),
        loss=DistillationLossConfig(
            alpha=0.6,
            temperature=4.0
        ),
        vocab_alignment=VocabularyAlignmentConfig(
            strategy="fuzzy_match",
            use_subword_matching=True
        )
    )


def load_distillation_config_from_yaml(config_path: str) -> DistillationConfig:
    """Load distillation configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract distillation section
    distill_config = config_dict.get('distillation', {})
    
    # Create configuration objects
    teacher_config = TeacherModelConfig(**distill_config.get('teacher_model', {}))
    loss_config = DistillationLossConfig(**distill_config.get('loss', {}))
    vocab_config = VocabularyAlignmentConfig(**distill_config.get('vocab_alignment', {}))
    
    # Create main config
    main_config = DistillationConfig(
        teacher_model=teacher_config,
        loss=loss_config,
        vocab_alignment=vocab_config,
        **{k: v for k, v in distill_config.items() 
           if k not in ['teacher_model', 'loss', 'vocab_alignment']}
    )
    
    main_config.validate()
    return main_config


def save_distillation_config_to_yaml(config: DistillationConfig, config_path: str):
    """Save distillation configuration to YAML file."""
    # Convert to dictionary
    config_dict = {
        'distillation': {
            'enabled': config.enabled,
            'teacher_model': {
                'model_name': config.teacher_model.model_name,
                'model_type': config.teacher_model.model_type,
                'device': config.teacher_model.device,
                'torch_dtype': config.teacher_model.torch_dtype,
                'temperature': config.teacher_model.temperature,
                'top_k': config.teacher_model.top_k,
                'top_p': config.teacher_model.top_p,
                'max_memory_gb': config.teacher_model.max_memory_gb,
                'batch_size': config.teacher_model.batch_size,
                'max_sequence_length': config.teacher_model.max_sequence_length,
                'logit_storage_path': config.teacher_model.logit_storage_path,
                'logit_format': config.teacher_model.logit_format,
                'logit_dtype': config.teacher_model.logit_dtype
            },
            'loss': {
                'alpha': config.loss.alpha,
                'temperature': config.loss.temperature,
                'use_hard_targets': config.loss.use_hard_targets,
                'use_soft_targets': config.loss.use_soft_targets,
                'adaptive_alpha': config.loss.adaptive_alpha,
                'alpha_schedule': config.loss.alpha_schedule,
                'alpha_min': config.loss.alpha_min,
                'curriculum_learning': config.loss.curriculum_learning
            },
            'vocab_alignment': {
                'strategy': config.vocab_alignment.strategy,
                'similarity_threshold': config.vocab_alignment.similarity_threshold,
                'use_subword_matching': config.vocab_alignment.use_subword_matching,
                'unk_token_strategy': config.vocab_alignment.unk_token_strategy
            },
            'distill_every_n_steps': config.distill_every_n_steps,
            'warmup_steps': config.warmup_steps,
            'extract_logits_offline': config.extract_logits_offline,
            'logit_cache_size': config.logit_cache_size,
            'log_distillation_metrics': config.log_distillation_metrics,
            'use_mixed_precision': config.use_mixed_precision
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Configuration registry for easy access
DISTILLATION_CONFIGS = {
    'conservative': get_conservative_distillation_config,
    'aggressive': get_aggressive_distillation_config,
    'character_level': get_character_level_distillation_config
}


def get_distillation_config(preset: str) -> DistillationConfig:
    """Get a predefined distillation configuration by name."""
    if preset not in DISTILLATION_CONFIGS:
        available = list(DISTILLATION_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    return DISTILLATION_CONFIGS[preset]()
