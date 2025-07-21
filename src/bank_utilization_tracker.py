#!/usr/bin/env python3
"""
Bank Utilization Tracker for MARU Continual Learning Framework

This module implements tracking of memory bank utilization and router confidence
to enable weighted sampling for VAE replay, replacing the catastrophic uniform
sampling strategy.

Based on research.md Section 2.2: "Generative Replay Instability"

Key Features:
1. Track memory bank access frequency during awake phase
2. Monitor router confidence/entropy for each bank selection
3. Provide utilization and confidence weights for biased VAE sampling
4. Replace uniform sampling with representative sampling

This addresses the critical flaw where "sampling equally from all four memory banks
represents a critical and likely catastrophic design flaw" by implementing
utilization-weighted and entropy-weighted sampling strategies.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class BankUtilizationTracker:
    """
    Tracks memory bank utilization and router confidence for weighted VAE sampling.
    
    This tracker monitors:
    1. Bank access frequency (how often each bank is selected)
    2. Router confidence (entropy of router decisions)
    3. Bank usage patterns over time
    4. Provides weights for biased sampling to replace uniform sampling
    """
    
    def __init__(self, 
                 num_banks: int = 4,
                 history_size: int = 1000,
                 confidence_threshold: float = 0.8,
                 utilization_decay: float = 0.99):
        """
        Initialize bank utilization tracker.
        
        Args:
            num_banks: Number of memory banks to track
            history_size: Size of rolling history for statistics
            confidence_threshold: Threshold for high-confidence decisions
            utilization_decay: Decay factor for utilization moving average
        """
        self.num_banks = num_banks
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.utilization_decay = utilization_decay
        
        # Bank access counters
        self.bank_access_counts = torch.zeros(num_banks)
        self.total_accesses = 0
        
        # Router confidence tracking
        self.router_entropy_history = deque(maxlen=history_size)
        self.bank_confidence_scores = torch.zeros(num_banks)
        
        # Utilization moving averages
        self.utilization_ema = torch.zeros(num_banks)
        
        # Detailed statistics
        self.access_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        logger.info(f"Initialized BankUtilizationTracker for {num_banks} banks")
    
    def update_bank_access(self, router_weights: torch.Tensor, step: Optional[int] = None) -> None:
        """
        Update bank access statistics from router weights.
        
        Args:
            router_weights: Router output weights (batch_size, num_banks) or (num_banks,)
            step: Optional step number for logging
        """
        if router_weights.dim() == 2:
            # Average across batch dimension
            avg_weights = router_weights.mean(dim=0)
        else:
            avg_weights = router_weights
        
        # Update access counts (weighted by router probabilities)
        self.bank_access_counts += avg_weights.detach().cpu()
        self.total_accesses += 1
        
        # Update utilization EMA
        self.utilization_ema = (self.utilization_decay * self.utilization_ema + 
                               (1 - self.utilization_decay) * avg_weights.detach().cpu())
        
        # Calculate router entropy (confidence measure)
        entropy = self._calculate_entropy(avg_weights)
        self.router_entropy_history.append(entropy)
        
        # Update bank-specific confidence scores
        self._update_confidence_scores(avg_weights, entropy)
        
        # Store detailed history
        self.access_history.append({
            'step': step,
            'weights': avg_weights.detach().cpu().clone(),
            'entropy': entropy,
            'timestamp': time.time()
        })
        
        if step is not None and step % 100 == 0:
            logger.debug(f"Step {step}: Bank utilization: {self.get_utilization_weights()}")
    
    def _calculate_entropy(self, weights: torch.Tensor) -> float:
        """Calculate entropy of router weights (lower = more confident)."""
        # Add small epsilon to avoid log(0)
        weights_safe = weights + 1e-8
        entropy = -torch.sum(weights_safe * torch.log(weights_safe)).item()
        return entropy
    
    def _update_confidence_scores(self, weights: torch.Tensor, entropy: float) -> None:
        """Update confidence scores for each bank based on selection patterns."""
        # High confidence = low entropy
        confidence = max(0.0, 1.0 - entropy / np.log(self.num_banks))  # Normalize by max entropy
        
        # Update confidence scores for banks that were selected
        for bank_idx in range(self.num_banks):
            bank_weight = weights[bank_idx].item()
            if bank_weight > 0.1:  # Bank was significantly selected
                # Update confidence score with exponential moving average
                current_confidence = self.bank_confidence_scores[bank_idx].item()
                self.bank_confidence_scores[bank_idx] = (
                    0.9 * current_confidence + 0.1 * confidence * bank_weight
                )
    
    def get_utilization_weights(self) -> torch.Tensor:
        """
        Get utilization-based weights for sampling.
        
        Returns:
            Normalized weights based on bank utilization frequency
        """
        if self.total_accesses == 0:
            # Return uniform weights if no data
            return torch.ones(self.num_banks) / self.num_banks
        
        # Use EMA utilization for more stable weights
        weights = self.utilization_ema.clone()
        
        # Normalize to sum to 1
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def get_confidence_weights(self) -> torch.Tensor:
        """
        Get confidence-based weights for sampling.
        
        Returns:
            Normalized weights based on router confidence for each bank
        """
        weights = self.bank_confidence_scores.clone()
        
        # Normalize to sum to 1
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def get_combined_weights(self, 
                           utilization_weight: float = 0.7,
                           confidence_weight: float = 0.3) -> torch.Tensor:
        """
        Get combined utilization and confidence weights.
        
        Args:
            utilization_weight: Weight for utilization component
            confidence_weight: Weight for confidence component
            
        Returns:
            Combined normalized weights for sampling
        """
        util_weights = self.get_utilization_weights()
        conf_weights = self.get_confidence_weights()
        
        combined = (utilization_weight * util_weights + 
                   confidence_weight * conf_weights)
        
        # Normalize
        combined = combined / (combined.sum() + 1e-8)
        
        return combined
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive utilization statistics."""
        util_weights = self.get_utilization_weights()
        conf_weights = self.get_confidence_weights()
        
        stats = {
            'total_accesses': self.total_accesses,
            'bank_access_counts': self.bank_access_counts.tolist(),
            'utilization_weights': util_weights.tolist(),
            'confidence_weights': conf_weights.tolist(),
            'utilization_ema': self.utilization_ema.tolist(),
            'confidence_scores': self.bank_confidence_scores.tolist(),
            'avg_router_entropy': np.mean(list(self.router_entropy_history)) if self.router_entropy_history else 0.0,
            'recent_entropy': list(self.router_entropy_history)[-10:] if self.router_entropy_history else []
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all tracking statistics."""
        self.bank_access_counts.zero_()
        self.total_accesses = 0
        self.utilization_ema.zero_()
        self.bank_confidence_scores.zero_()
        self.router_entropy_history.clear()
        self.access_history.clear()
        self.confidence_history.clear()
        
        logger.info("Reset bank utilization statistics")
    
    def should_use_weighted_sampling(self) -> bool:
        """
        Determine if weighted sampling should be used based on utilization patterns.
        
        Returns:
            True if utilization is sufficiently uneven to warrant weighted sampling
        """
        if self.total_accesses < 10:  # Need minimum data
            return False
        
        util_weights = self.get_utilization_weights()
        
        # Check if utilization is significantly non-uniform
        # Calculate coefficient of variation
        mean_util = util_weights.mean().item()
        std_util = util_weights.std().item()
        
        if mean_util > 0:
            cv = std_util / mean_util
            # Use weighted sampling if coefficient of variation > 0.5
            return cv > 0.5
        
        return False


def create_bank_utilization_tracker_from_config(config: Dict[str, Any]) -> BankUtilizationTracker:
    """
    Create BankUtilizationTracker from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BankUtilizationTracker instance
    """
    tracker_config = config.get('bank_utilization_tracker', {})
    
    num_banks = tracker_config.get('num_banks', 4)
    history_size = tracker_config.get('history_size', 1000)
    confidence_threshold = tracker_config.get('confidence_threshold', 0.8)
    utilization_decay = tracker_config.get('utilization_decay', 0.99)
    
    return BankUtilizationTracker(
        num_banks=num_banks,
        history_size=history_size,
        confidence_threshold=confidence_threshold,
        utilization_decay=utilization_decay
    )
