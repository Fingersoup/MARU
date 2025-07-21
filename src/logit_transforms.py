#!/usr/bin/env python3
"""
Logit Transformation for MARU Knowledge Distillation

This module implements logit transformation and reshaping utilities for knowledge
distillation between teacher models and MARU's character-level architecture.

Key Features:
- Teacher logit reshaping to match MARU vocab size (256)
- Logit interpolation for vocabulary mismatches
- Support for multiple teacher vocabularies
- Temperature scaling and normalization
- Batch processing with memory efficiency

Required for multi-target distillation in Phase 0.2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import math

logger = logging.getLogger(__name__)

# MARU constants
MARU_VOCAB_SIZE = 256


class LogitTransformer:
    """
    Handles logit transformation and reshaping for knowledge distillation.
    
    This class provides methods to transform teacher model logits to match
    the student model's vocabulary and output dimensions.
    """
    
    def __init__(self, 
                 teacher_vocab_size: int,
                 student_vocab_size: int = MARU_VOCAB_SIZE,
                 interpolation_method: str = "linear",
                 temperature: float = 1.0):
        """
        Initialize the logit transformer.
        
        Args:
            teacher_vocab_size: Size of teacher model vocabulary
            student_vocab_size: Size of student model vocabulary
            interpolation_method: Method for logit interpolation ("linear", "nearest", "cubic")
            temperature: Temperature for softmax scaling
        """
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        self.interpolation_method = interpolation_method
        self.temperature = temperature
        
        # Pre-compute interpolation indices for efficiency
        self.interpolation_indices = self._compute_interpolation_indices()
        
        logger.info(f"LogitTransformer initialized: {teacher_vocab_size} -> {student_vocab_size}, "
                   f"method={interpolation_method}, temperature={temperature}")
    
    def _compute_interpolation_indices(self) -> torch.Tensor:
        """
        Pre-compute interpolation indices for efficient logit transformation.
        
        Returns:
            Interpolation indices tensor
        """
        if self.teacher_vocab_size == self.student_vocab_size:
            # No transformation needed
            return torch.arange(self.student_vocab_size)
        
        # Create mapping from student indices to teacher indices
        teacher_indices = torch.linspace(0, self.teacher_vocab_size - 1, self.student_vocab_size)
        
        if self.interpolation_method == "nearest":
            return teacher_indices.round().long()
        else:
            return teacher_indices
    
    def transform_logits(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Transform teacher logits to match student vocabulary size.
        
        Args:
            teacher_logits: Teacher logits of shape (..., teacher_vocab_size)
            
        Returns:
            Transformed logits of shape (..., student_vocab_size)
        """
        if teacher_logits.size(-1) != self.teacher_vocab_size:
            raise ValueError(f"Expected teacher logits size {self.teacher_vocab_size}, "
                           f"got {teacher_logits.size(-1)}")
        
        if self.teacher_vocab_size == self.student_vocab_size:
            # No transformation needed
            return teacher_logits / self.temperature
        
        # Move interpolation indices to same device
        indices = self.interpolation_indices.to(teacher_logits.device)
        
        if self.interpolation_method == "linear":
            transformed_logits = self._linear_interpolation(teacher_logits, indices)
        elif self.interpolation_method == "nearest":
            transformed_logits = self._nearest_interpolation(teacher_logits, indices)
        elif self.interpolation_method == "cubic":
            transformed_logits = self._cubic_interpolation(teacher_logits, indices)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
        
        # Apply temperature scaling
        return transformed_logits / self.temperature
    
    def _linear_interpolation(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Perform linear interpolation of logits.
        
        Args:
            logits: Input logits
            indices: Interpolation indices
            
        Returns:
            Interpolated logits
        """
        # Get integer and fractional parts
        indices_floor = indices.floor().long()
        indices_ceil = indices.ceil().long()
        weights = indices - indices_floor.float()
        
        # Clamp indices to valid range
        indices_floor = torch.clamp(indices_floor, 0, self.teacher_vocab_size - 1)
        indices_ceil = torch.clamp(indices_ceil, 0, self.teacher_vocab_size - 1)
        
        # Gather values and interpolate
        logits_floor = torch.gather(logits, -1, indices_floor.expand_as(logits[..., :len(indices_floor)]))
        logits_ceil = torch.gather(logits, -1, indices_ceil.expand_as(logits[..., :len(indices_ceil)]))
        
        # Linear interpolation
        weights = weights.to(logits.device).unsqueeze(0).expand_as(logits_floor)
        interpolated = logits_floor * (1 - weights) + logits_ceil * weights
        
        return interpolated
    
    def _nearest_interpolation(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Perform nearest neighbor interpolation of logits.
        
        Args:
            logits: Input logits
            indices: Interpolation indices (already rounded)
            
        Returns:
            Interpolated logits
        """
        # Clamp indices to valid range
        indices = torch.clamp(indices, 0, self.teacher_vocab_size - 1)
        
        # Gather values
        return torch.gather(logits, -1, indices.expand_as(logits[..., :len(indices)]))
    
    def _cubic_interpolation(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Perform cubic interpolation of logits (simplified version).
        
        Args:
            logits: Input logits
            indices: Interpolation indices
            
        Returns:
            Interpolated logits
        """
        # For simplicity, fall back to linear interpolation
        # A full cubic implementation would require more complex indexing
        return self._linear_interpolation(logits, indices)
    
    def transform_probabilities(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        Transform teacher probabilities to match student vocabulary size.
        
        Args:
            teacher_probs: Teacher probabilities of shape (..., teacher_vocab_size)
            
        Returns:
            Transformed probabilities of shape (..., student_vocab_size)
        """
        # Convert to logits, transform, then back to probabilities
        teacher_logits = torch.log(teacher_probs + 1e-8)  # Add small epsilon for numerical stability
        transformed_logits = self.transform_logits(teacher_logits)
        transformed_probs = F.softmax(transformed_logits, dim=-1)
        
        return transformed_probs
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update the temperature for softmax scaling.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        logger.debug(f"Updated temperature to {temperature}")


class MultiTeacherLogitTransformer:
    """
    Handles logit transformation for multiple teacher models with different vocabularies.
    
    This class manages multiple LogitTransformer instances and provides
    ensemble methods for combining multiple teacher outputs.
    """
    
    def __init__(self, 
                 teacher_vocab_sizes: List[int],
                 student_vocab_size: int = MARU_VOCAB_SIZE,
                 ensemble_method: str = "average",
                 teacher_weights: Optional[List[float]] = None):
        """
        Initialize the multi-teacher logit transformer.
        
        Args:
            teacher_vocab_sizes: List of teacher vocabulary sizes
            student_vocab_size: Student vocabulary size
            ensemble_method: Method for combining teacher outputs ("average", "weighted", "max")
            teacher_weights: Weights for weighted ensemble (if None, uses uniform weights)
        """
        self.teacher_vocab_sizes = teacher_vocab_sizes
        self.student_vocab_size = student_vocab_size
        self.ensemble_method = ensemble_method
        
        # Initialize individual transformers
        self.transformers = [
            LogitTransformer(vocab_size, student_vocab_size)
            for vocab_size in teacher_vocab_sizes
        ]
        
        # Set up ensemble weights
        if teacher_weights is None:
            self.teacher_weights = [1.0 / len(teacher_vocab_sizes)] * len(teacher_vocab_sizes)
        else:
            if len(teacher_weights) != len(teacher_vocab_sizes):
                raise ValueError("Number of weights must match number of teachers")
            # Normalize weights
            total_weight = sum(teacher_weights)
            self.teacher_weights = [w / total_weight for w in teacher_weights]
        
        logger.info(f"MultiTeacherLogitTransformer initialized with {len(teacher_vocab_sizes)} teachers")
    
    def transform_multi_teacher_logits(self, teacher_logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Transform and combine logits from multiple teachers.
        
        Args:
            teacher_logits_list: List of teacher logits tensors
            
        Returns:
            Combined transformed logits
        """
        if len(teacher_logits_list) != len(self.transformers):
            raise ValueError(f"Expected {len(self.transformers)} teacher logits, "
                           f"got {len(teacher_logits_list)}")
        
        # Transform each teacher's logits
        transformed_logits = []
        for i, (teacher_logits, transformer) in enumerate(zip(teacher_logits_list, self.transformers)):
            transformed = transformer.transform_logits(teacher_logits)
            transformed_logits.append(transformed)
        
        # Combine using ensemble method
        if self.ensemble_method == "average":
            combined_logits = torch.stack(transformed_logits).mean(dim=0)
        elif self.ensemble_method == "weighted":
            weighted_logits = []
            for i, logits in enumerate(transformed_logits):
                weighted_logits.append(logits * self.teacher_weights[i])
            combined_logits = torch.stack(weighted_logits).sum(dim=0)
        elif self.ensemble_method == "max":
            combined_logits = torch.stack(transformed_logits).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return combined_logits


def create_logit_transformer(teacher_vocab_size: int,
                            student_vocab_size: int = MARU_VOCAB_SIZE,
                            interpolation_method: str = "linear",
                            temperature: float = 1.0) -> LogitTransformer:
    """
    Factory function to create a logit transformer.
    
    Args:
        teacher_vocab_size: Size of teacher vocabulary
        student_vocab_size: Size of student vocabulary
        interpolation_method: Interpolation method
        temperature: Temperature for scaling
        
    Returns:
        Configured LogitTransformer instance
    """
    return LogitTransformer(teacher_vocab_size, student_vocab_size, 
                          interpolation_method, temperature)


def validate_logit_transformation(original_logits: torch.Tensor,
                                transformed_logits: torch.Tensor,
                                tolerance: float = 1e-6) -> bool:
    """
    Validate that logit transformation preserves essential properties.
    
    Args:
        original_logits: Original teacher logits
        transformed_logits: Transformed logits
        tolerance: Numerical tolerance
        
    Returns:
        True if transformation is valid
    """
    try:
        # Check that probabilities are valid
        original_probs = F.softmax(original_logits, dim=-1)
        transformed_probs = F.softmax(transformed_logits, dim=-1)
        
        # Check probability sums
        orig_sum = original_probs.sum(dim=-1)
        trans_sum = transformed_probs.sum(dim=-1)
        
        orig_valid = torch.allclose(orig_sum, torch.ones_like(orig_sum), atol=tolerance)
        trans_valid = torch.allclose(trans_sum, torch.ones_like(trans_sum), atol=tolerance)
        
        # Check for NaN or Inf values
        no_nan_orig = not torch.isnan(original_logits).any()
        no_nan_trans = not torch.isnan(transformed_logits).any()
        no_inf_orig = not torch.isinf(original_logits).any()
        no_inf_trans = not torch.isinf(transformed_logits).any()
        
        return orig_valid and trans_valid and no_nan_orig and no_nan_trans and no_inf_orig and no_inf_trans
        
    except Exception as e:
        logger.error(f"Error validating logit transformation: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test single teacher transformation
    teacher_vocab_size = 50257  # GPT-2
    student_vocab_size = 256    # MARU
    
    transformer = create_logit_transformer(teacher_vocab_size, student_vocab_size)
    
    # Test transformation
    batch_size = 4
    teacher_logits = torch.randn(batch_size, teacher_vocab_size)
    transformed_logits = transformer.transform_logits(teacher_logits)
    
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"Transformed logits shape: {transformed_logits.shape}")
    
    # Validate transformation
    is_valid = validate_logit_transformation(teacher_logits, transformed_logits)
    print(f"Transformation valid: {is_valid}")
    
    # Test multi-teacher transformation
    teacher_vocab_sizes = [50257, 32000, 30522]  # GPT-2, T5, BERT
    multi_transformer = MultiTeacherLogitTransformer(teacher_vocab_sizes, student_vocab_size)
    
    teacher_logits_list = [torch.randn(batch_size, size) for size in teacher_vocab_sizes]
    combined_logits = multi_transformer.transform_multi_teacher_logits(teacher_logits_list)
    
    print(f"Combined logits shape: {combined_logits.shape}")
    print("Multi-teacher transformation completed successfully!")
