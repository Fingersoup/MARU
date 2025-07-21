#!/usr/bin/env python3
"""
Adaptive Temperature Scaling for Knowledge Distillation

This module implements research-backed adaptive temperature scaling methods for
optimizing knowledge transfer during MARU consolidation. Based on Section 3.3.2
of the research document (catacombs/tacos.md).

Key Methods Implemented:
1. Curriculum Temperature (CTKD): Learnable temperature with gradient reversal
2. Sharpness-based Adaptation (ATKD): Dynamic adjustment based on distribution mismatch
3. Logit Correlation Adaptation: Temperature based on teacher's max logit value

Research References:
- Section 3.3.2 "Adaptive Temperature Scaling"
- Curriculum Temperature Knowledge Distillation (CTKD)
- Adaptive Temperature Knowledge Distillation (ATKD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TemperatureAdaptationMethod(Enum):
    """Enumeration of available temperature adaptation methods."""
    FIXED = "fixed"
    CURRICULUM = "curriculum"
    SHARPNESS_BASED = "sharpness_based"
    LOGIT_CORRELATION = "logit_correlation"
    COMBINED = "combined"


@dataclass
class AdaptiveTemperatureConfig:
    """Configuration for adaptive temperature scaling."""
    
    # Method selection
    method: TemperatureAdaptationMethod = TemperatureAdaptationMethod.SHARPNESS_BASED
    
    # Base temperature settings
    initial_temperature: float = 3.0
    min_temperature: float = 1.0
    max_temperature: float = 8.0
    
    # Curriculum temperature settings
    curriculum_steps: int = 1000
    curriculum_lr: float = 0.01
    gradient_reversal_lambda: float = 1.0
    
    # Sharpness-based adaptation settings
    target_sharpness_gap: float = 0.1
    sharpness_adaptation_rate: float = 0.1
    sharpness_smoothing_factor: float = 0.9
    
    # Logit correlation settings
    correlation_sensitivity: float = 0.5
    max_logit_threshold: float = 10.0
    
    # Combined method weights
    curriculum_weight: float = 0.3
    sharpness_weight: float = 0.4
    correlation_weight: float = 0.3
    
    # Monitoring and logging
    log_temperature_changes: bool = True
    temperature_change_threshold: float = 0.1


class AdaptiveTemperatureScaler:
    """
    Adaptive temperature scaling system for knowledge distillation.
    
    Implements multiple research-backed methods for dynamically adjusting
    the distillation temperature during consolidation to optimize knowledge transfer.
    """
    
    def __init__(self, config: AdaptiveTemperatureConfig):
        """
        Initialize the adaptive temperature scaler.
        
        Args:
            config: Configuration for temperature adaptation
        """
        self.config = config
        self.current_temperature = config.initial_temperature
        self.step_count = 0
        
        # Initialize method-specific components
        self._init_curriculum_components()
        self._init_sharpness_components()
        self._init_correlation_components()
        
        # Monitoring
        self.temperature_history = []
        self.last_logged_temperature = config.initial_temperature
        
        logger.info(f"AdaptiveTemperatureScaler initialized with method: {config.method.value}")
        logger.info(f"Initial temperature: {config.initial_temperature}")
    
    def _init_curriculum_components(self):
        """Initialize curriculum temperature components."""
        if self.config.method in [TemperatureAdaptationMethod.CURRICULUM, TemperatureAdaptationMethod.COMBINED]:
            # Learnable temperature parameter
            self.learnable_temperature = nn.Parameter(torch.tensor(self.config.initial_temperature))
            self.temperature_optimizer = torch.optim.Adam([self.learnable_temperature], lr=self.config.curriculum_lr)
            logger.debug("Curriculum temperature components initialized")
    
    def _init_sharpness_components(self):
        """Initialize sharpness-based adaptation components."""
        if self.config.method in [TemperatureAdaptationMethod.SHARPNESS_BASED, TemperatureAdaptationMethod.COMBINED]:
            self.teacher_sharpness_ema = None
            self.student_sharpness_ema = None
            self.sharpness_gap_ema = None
            logger.debug("Sharpness-based adaptation components initialized")
    
    def _init_correlation_components(self):
        """Initialize logit correlation components."""
        if self.config.method in [TemperatureAdaptationMethod.LOGIT_CORRELATION, TemperatureAdaptationMethod.COMBINED]:
            self.max_logit_ema = None
            logger.debug("Logit correlation components initialized")
    
    def update_temperature(self, 
                          student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          distillation_loss: Optional[torch.Tensor] = None) -> float:
        """
        Update the temperature based on the configured adaptation method.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            distillation_loss: Current distillation loss (for curriculum method)
            
        Returns:
            Updated temperature value
        """
        self.step_count += 1
        
        if self.config.method == TemperatureAdaptationMethod.FIXED:
            # No adaptation - return initial temperature
            return self.config.initial_temperature
        
        elif self.config.method == TemperatureAdaptationMethod.CURRICULUM:
            new_temp = self._update_curriculum_temperature(distillation_loss)
        
        elif self.config.method == TemperatureAdaptationMethod.SHARPNESS_BASED:
            new_temp = self._update_sharpness_based_temperature(student_logits, teacher_logits)
        
        elif self.config.method == TemperatureAdaptationMethod.LOGIT_CORRELATION:
            new_temp = self._update_correlation_based_temperature(teacher_logits)
        
        elif self.config.method == TemperatureAdaptationMethod.COMBINED:
            new_temp = self._update_combined_temperature(student_logits, teacher_logits, distillation_loss)
        
        else:
            raise ValueError(f"Unknown temperature adaptation method: {self.config.method}")
        
        # Clamp temperature to valid range
        new_temp = torch.clamp(torch.tensor(new_temp), 
                              self.config.min_temperature, 
                              self.config.max_temperature).item()
        
        # Update current temperature and log if significant change
        if abs(new_temp - self.current_temperature) > self.config.temperature_change_threshold:
            if self.config.log_temperature_changes:
                logger.info(f"Temperature adapted: {self.current_temperature:.3f} → {new_temp:.3f} "
                           f"(method: {self.config.method.value}, step: {self.step_count})")
            self.last_logged_temperature = new_temp
        
        self.current_temperature = new_temp
        self.temperature_history.append(new_temp)
        
        return new_temp
    
    def _update_curriculum_temperature(self, distillation_loss: Optional[torch.Tensor]) -> float:
        """Update temperature using curriculum learning approach (CTKD)."""
        if distillation_loss is None:
            return self.current_temperature
        
        # Gradient reversal: maximize distillation loss to make task harder
        reversed_loss = -distillation_loss * self.config.gradient_reversal_lambda
        
        # Update learnable temperature
        self.temperature_optimizer.zero_grad()
        reversed_loss.backward(retain_graph=True)
        self.temperature_optimizer.step()
        
        return self.learnable_temperature.item()
    
    def _update_sharpness_based_temperature(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
        """Update temperature using sharpness-based adaptation (ATKD)."""
        # Calculate sharpness (log-sum-exp of logits)
        teacher_sharpness = torch.logsumexp(teacher_logits, dim=-1).mean().item()
        student_sharpness = torch.logsumexp(student_logits, dim=-1).mean().item()
        
        # Update exponential moving averages
        if self.teacher_sharpness_ema is None:
            self.teacher_sharpness_ema = teacher_sharpness
            self.student_sharpness_ema = student_sharpness
        else:
            alpha = self.config.sharpness_smoothing_factor
            self.teacher_sharpness_ema = alpha * self.teacher_sharpness_ema + (1 - alpha) * teacher_sharpness
            self.student_sharpness_ema = alpha * self.student_sharpness_ema + (1 - alpha) * student_sharpness
        
        # Calculate sharpness gap
        sharpness_gap = abs(self.teacher_sharpness_ema - self.student_sharpness_ema)
        
        # Update sharpness gap EMA
        if self.sharpness_gap_ema is None:
            self.sharpness_gap_ema = sharpness_gap
        else:
            self.sharpness_gap_ema = (self.config.sharpness_smoothing_factor * self.sharpness_gap_ema + 
                                     (1 - self.config.sharpness_smoothing_factor) * sharpness_gap)
        
        # Adapt temperature to minimize sharpness gap
        gap_error = self.sharpness_gap_ema - self.config.target_sharpness_gap
        temperature_adjustment = gap_error * self.config.sharpness_adaptation_rate
        
        return self.current_temperature + temperature_adjustment
    
    def _update_correlation_based_temperature(self, teacher_logits: torch.Tensor) -> float:
        """Update temperature using logit correlation adaptation."""
        # Calculate teacher's maximum logit value
        max_logit = teacher_logits.max(dim=-1)[0].mean().item()
        
        # Update exponential moving average
        if self.max_logit_ema is None:
            self.max_logit_ema = max_logit
        else:
            alpha = self.config.sharpness_smoothing_factor
            self.max_logit_ema = alpha * self.max_logit_ema + (1 - alpha) * max_logit
        
        # Adapt temperature based on max logit value
        # Higher max logits suggest more confident predictions, may need higher temperature
        normalized_max_logit = min(self.max_logit_ema / self.config.max_logit_threshold, 1.0)
        temperature_factor = 1.0 + normalized_max_logit * self.config.correlation_sensitivity
        
        return self.config.initial_temperature * temperature_factor
    
    def _update_combined_temperature(self, 
                                   student_logits: torch.Tensor,
                                   teacher_logits: torch.Tensor,
                                   distillation_loss: Optional[torch.Tensor]) -> float:
        """Update temperature using combined approach."""
        # Get individual method temperatures
        curriculum_temp = self._update_curriculum_temperature(distillation_loss) if distillation_loss else self.config.initial_temperature
        sharpness_temp = self._update_sharpness_based_temperature(student_logits, teacher_logits)
        correlation_temp = self._update_correlation_based_temperature(teacher_logits)
        
        # Weighted combination
        combined_temp = (self.config.curriculum_weight * curriculum_temp +
                        self.config.sharpness_weight * sharpness_temp +
                        self.config.correlation_weight * correlation_temp)
        
        return combined_temp
    
    def get_current_temperature(self) -> float:
        """Get the current temperature value."""
        return self.current_temperature
    
    def get_temperature_statistics(self) -> Dict[str, Any]:
        """Get statistics about temperature adaptation."""
        if not self.temperature_history:
            return {
                "current_temperature": self.current_temperature,
                "initial_temperature": self.config.initial_temperature,
                "mean_temperature": self.current_temperature,
                "std_temperature": 0.0,
                "min_temperature": self.current_temperature,
                "max_temperature": self.current_temperature,
                "total_steps": self.step_count,
                "adaptation_method": self.config.method.value
            }

        history = torch.tensor(self.temperature_history)
        return {
            "current_temperature": self.current_temperature,
            "initial_temperature": self.config.initial_temperature,
            "mean_temperature": history.mean().item(),
            "std_temperature": history.std().item() if len(self.temperature_history) > 1 else 0.0,
            "min_temperature": history.min().item(),
            "max_temperature": history.max().item(),
            "total_steps": self.step_count,
            "adaptation_method": self.config.method.value
        }
    
    def reset(self):
        """Reset the temperature scaler to initial state."""
        self.current_temperature = self.config.initial_temperature
        self.step_count = 0
        self.temperature_history = []
        self.last_logged_temperature = self.config.initial_temperature
        
        # Reset method-specific components
        if hasattr(self, 'learnable_temperature'):
            self.learnable_temperature.data.fill_(self.config.initial_temperature)
        
        self.teacher_sharpness_ema = None
        self.student_sharpness_ema = None
        self.sharpness_gap_ema = None
        self.max_logit_ema = None
        
        logger.info("AdaptiveTemperatureScaler reset to initial state")
