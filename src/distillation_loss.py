"""
Knowledge Distillation Loss Functions

This module implements various loss functions for knowledge distillation between
teacher and student models, including temperature-scaled KL divergence, attention
transfer, and feature matching losses.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
import logging

logger = logging.getLogger(__name__)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 3.0,
    reduction: str = "batchmean"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute knowledge distillation loss combining soft and hard targets.
    
    Args:
        student_logits: Student model logits (batch_size, vocab_size)
        teacher_logits: Teacher model logits (batch_size, vocab_size)
        hard_targets: Ground truth targets (batch_size,)
        alpha: Weight for distillation loss (0.0 = only hard targets, 1.0 = only soft targets)
        temperature: Temperature for softmax scaling
        reduction: Reduction method for KL divergence
        
    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    # Ensure tensors are on the same device
    device = student_logits.device
    teacher_logits = teacher_logits.to(device)
    hard_targets = hard_targets.to(device)
    
    # Compute soft target loss (KL divergence with temperature scaling)
    if alpha > 0:
        soft_loss = compute_kl_divergence_loss(
            student_logits, teacher_logits, temperature, reduction
        )
    else:
        soft_loss = torch.tensor(0.0, device=device)
    
    # Compute hard target loss (cross-entropy)
    if alpha < 1.0:
        hard_loss = F.cross_entropy(student_logits, hard_targets, reduction='mean')
    else:
        hard_loss = torch.tensor(0.0, device=device)
    
    # Combine losses
    total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
    
    # Prepare loss components for logging
    loss_components = {
        'total_loss': total_loss.item(),
        'soft_loss': soft_loss.item(),
        'hard_loss': hard_loss.item(),
        'alpha': alpha,
        'temperature': temperature
    }
    
    return total_loss, loss_components


def compute_kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 3.0,
    reduction: str = "batchmean"
) -> torch.Tensor:
    """
    Compute KL divergence loss with temperature scaling and numerical safeguarding.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softmax scaling
        reduction: Reduction method

    Returns:
        KL divergence loss scaled by temperature squared
    """
    epsilon = 1e-8  # Numerical safeguarding

    # Apply temperature scaling
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Add epsilon to prevent log(0) issues
    teacher_probs = teacher_probs + epsilon
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)  # Renormalize

    # Compute KL divergence
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)

    # Scale by temperature squared (standard in knowledge distillation)
    return kl_loss * (temperature ** 2)


def compute_mse_logit_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 3.0
) -> torch.Tensor:
    """
    Compute MSE loss on raw logits (more numerically stable than KL divergence).

    Based on research showing that MSE on logits can outperform KL divergence
    and is more robust to extreme values that cause NaNs.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature parameter (not used for MSE, kept for API consistency)

    Returns:
        MSE loss between raw logits
    """
    # Direct MSE on raw logits - no temperature scaling needed
    # This avoids the log/exp operations that can cause numerical instability
    mse_loss = F.mse_loss(student_logits, teacher_logits, reduction='mean')

    return mse_loss


def compute_attention_transfer_loss(
    student_attentions: List[torch.Tensor],
    teacher_attentions: List[torch.Tensor],
    layer_mapping: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Compute attention transfer loss between teacher and student attention weights.
    
    Args:
        student_attentions: List of student attention tensors
        teacher_attentions: List of teacher attention tensors
        layer_mapping: Mapping from student layers to teacher layers
        
    Returns:
        Attention transfer loss
    """
    if not student_attentions or not teacher_attentions:
        return torch.tensor(0.0)
    
    # Default layer mapping (align layers proportionally)
    if layer_mapping is None:
        num_student_layers = len(student_attentions)
        num_teacher_layers = len(teacher_attentions)
        layer_mapping = [
            int(i * num_teacher_layers / num_student_layers)
            for i in range(num_student_layers)
        ]
    
    total_loss = 0.0
    num_pairs = 0
    
    for student_idx, teacher_idx in enumerate(layer_mapping):
        if teacher_idx < len(teacher_attentions):
            student_att = student_attentions[student_idx]
            teacher_att = teacher_attentions[teacher_idx]
            
            # Normalize attention weights
            student_att_norm = F.normalize(student_att.view(student_att.size(0), -1), p=2, dim=1)
            teacher_att_norm = F.normalize(teacher_att.view(teacher_att.size(0), -1), p=2, dim=1)
            
            # Compute MSE loss between normalized attention weights
            att_loss = F.mse_loss(student_att_norm, teacher_att_norm)
            total_loss += att_loss
            num_pairs += 1
    
    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


def compute_feature_matching_loss(
    student_features: List[torch.Tensor],
    teacher_features: List[torch.Tensor],
    layer_mapping: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Compute feature matching loss between intermediate representations.
    
    Args:
        student_features: List of student feature tensors
        teacher_features: List of teacher feature tensors
        layer_mapping: Mapping from student layers to teacher layers
        
    Returns:
        Feature matching loss
    """
    if not student_features or not teacher_features:
        return torch.tensor(0.0)
    
    # Default layer mapping
    if layer_mapping is None:
        num_student_layers = len(student_features)
        num_teacher_layers = len(teacher_features)
        layer_mapping = [
            int(i * num_teacher_layers / num_student_layers)
            for i in range(num_student_layers)
        ]
    
    total_loss = 0.0
    num_pairs = 0
    
    for student_idx, teacher_idx in enumerate(layer_mapping):
        if teacher_idx < len(teacher_features):
            student_feat = student_features[student_idx]
            teacher_feat = teacher_features[teacher_idx]
            
            # Handle dimension mismatch with linear projection
            if student_feat.size(-1) != teacher_feat.size(-1):
                # Simple mean pooling for dimension reduction
                if student_feat.size(-1) > teacher_feat.size(-1):
                    # Reduce student features
                    factor = student_feat.size(-1) // teacher_feat.size(-1)
                    student_feat = student_feat.view(*student_feat.shape[:-1], -1, factor).mean(-1)
                else:
                    # Reduce teacher features
                    factor = teacher_feat.size(-1) // student_feat.size(-1)
                    teacher_feat = teacher_feat.view(*teacher_feat.shape[:-1], -1, factor).mean(-1)
            
            # Compute MSE loss between features
            feat_loss = F.mse_loss(student_feat, teacher_feat)
            total_loss += feat_loss
            num_pairs += 1
    
    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


def compute_adaptive_alpha(
    current_step: int,
    total_steps: int,
    initial_alpha: float = 0.7,
    final_alpha: float = 0.1,
    schedule: str = "linear"
) -> float:
    """
    Compute adaptive alpha value for distillation loss weighting.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        initial_alpha: Initial alpha value
        final_alpha: Final alpha value
        schedule: Schedule type ("linear", "cosine", "exponential")
        
    Returns:
        Adaptive alpha value
    """
    if current_step >= total_steps:
        return final_alpha
    
    progress = current_step / total_steps
    
    if schedule == "linear":
        alpha = initial_alpha + (final_alpha - initial_alpha) * progress
    elif schedule == "cosine":
        alpha = final_alpha + (initial_alpha - final_alpha) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == "exponential":
        decay_rate = math.log(final_alpha / initial_alpha)
        alpha = initial_alpha * math.exp(decay_rate * progress)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return max(0.0, min(1.0, alpha))


def compute_curriculum_temperature(
    current_step: int,
    curriculum_steps: int,
    initial_temp: float = 5.0,
    final_temp: float = 3.0,
    schedule: str = "linear"
) -> float:
    """
    Compute curriculum temperature for progressive distillation.
    
    Args:
        current_step: Current training step
        curriculum_steps: Steps for curriculum learning
        initial_temp: Initial temperature (higher = softer)
        final_temp: Final temperature
        schedule: Schedule type
        
    Returns:
        Curriculum temperature value
    """
    if current_step >= curriculum_steps:
        return final_temp
    
    progress = current_step / curriculum_steps
    
    if schedule == "linear":
        temp = initial_temp + (final_temp - initial_temp) * progress
    elif schedule == "exponential":
        decay_rate = math.log(final_temp / initial_temp)
        temp = initial_temp * math.exp(decay_rate * progress)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    return max(1.0, temp)  # Ensure temperature is at least 1.0


def compute_teacher_student_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Compute agreement metrics between teacher and student predictions.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        top_k: Number of top predictions to consider
        
    Returns:
        Dictionary of agreement metrics
    """
    # Get top-k predictions
    student_top_k = torch.topk(student_logits, top_k, dim=-1).indices
    teacher_top_k = torch.topk(teacher_logits, top_k, dim=-1).indices
    
    # Top-1 agreement
    student_top1 = student_logits.argmax(dim=-1)
    teacher_top1 = teacher_logits.argmax(dim=-1)
    top1_agreement = (student_top1 == teacher_top1).float().mean().item()
    
    # Top-k agreement
    batch_size = student_logits.size(0)
    topk_agreement = 0.0
    
    for i in range(batch_size):
        student_set = set(student_top_k[i].cpu().numpy())
        teacher_set = set(teacher_top_k[i].cpu().numpy())
        intersection = len(student_set.intersection(teacher_set))
        topk_agreement += intersection / top_k
    
    topk_agreement /= batch_size
    
    # KL divergence (without temperature scaling)
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    kl_div = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs,
        reduction='batchmean'
    ).item()
    
    # Jensen-Shannon divergence
    m = 0.5 * (student_probs + teacher_probs)
    js_div = 0.5 * F.kl_div(F.log_softmax(student_logits, dim=-1), m, reduction='batchmean') + \
             0.5 * F.kl_div(F.log_softmax(teacher_logits, dim=-1), m, reduction='batchmean')
    js_div = js_div.item()
    
    return {
        'top1_agreement': top1_agreement,
        f'top{top_k}_agreement': topk_agreement,
        'kl_divergence': kl_div,
        'js_divergence': js_div
    }


class DistillationLossCalculator:
    """
    Comprehensive distillation loss calculator with support for multiple loss types
    and adaptive scheduling.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 3.0,
        use_attention_transfer: bool = False,
        use_feature_matching: bool = False,
        adaptive_alpha: bool = False,
        curriculum_learning: bool = False
    ):
        self.alpha = alpha
        self.temperature = temperature
        self.use_attention_transfer = use_attention_transfer
        self.use_feature_matching = use_feature_matching
        self.adaptive_alpha = adaptive_alpha
        self.curriculum_learning = curriculum_learning
        
        self.step_count = 0
        self.total_steps = None
    
    def set_training_schedule(self, total_steps: int):
        """Set the total number of training steps for adaptive scheduling."""
        self.total_steps = total_steps
    
    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None,
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute comprehensive distillation loss.
        
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Update step count
        self.step_count += 1
        
        # Compute adaptive parameters
        current_alpha = self.alpha
        current_temp = self.temperature
        
        if self.adaptive_alpha and self.total_steps:
            current_alpha = compute_adaptive_alpha(
                self.step_count, self.total_steps, self.alpha, 0.1
            )
        
        if self.curriculum_learning and self.total_steps:
            current_temp = compute_curriculum_temperature(
                self.step_count, self.total_steps // 2, 5.0, self.temperature
            )
        
        # Compute main distillation loss
        main_loss, loss_components = compute_distillation_loss(
            student_logits, teacher_logits, hard_targets,
            current_alpha, current_temp
        )
        
        total_loss = main_loss
        
        # Add attention transfer loss
        if self.use_attention_transfer and student_attentions and teacher_attentions:
            att_loss = compute_attention_transfer_loss(student_attentions, teacher_attentions)
            total_loss += 0.1 * att_loss  # Weight attention loss
            loss_components['attention_loss'] = att_loss.item()
        
        # Add feature matching loss
        if self.use_feature_matching and student_features and teacher_features:
            feat_loss = compute_feature_matching_loss(student_features, teacher_features)
            total_loss += 0.1 * feat_loss  # Weight feature loss
            loss_components['feature_loss'] = feat_loss.item()
        
        # Compute agreement metrics
        agreement_metrics = compute_teacher_student_agreement(student_logits, teacher_logits)
        loss_components.update(agreement_metrics)
        
        # Update loss components with current parameters
        loss_components.update({
            'current_alpha': current_alpha,
            'current_temperature': current_temp,
            'step_count': self.step_count
        })
        
        return total_loss, loss_components
