#!/usr/bin/env python3
"""
Memory Bank Specialization Monitoring for MARU Continual Learning Framework

This module provides comprehensive monitoring of memory bank specialization including:
- Cosine similarity tracking between memory banks
- Orthogonality regularization loss monitoring
- Bank-specific usage statistics
- Specialization drift detection

Part of Phase 4.5: Enhanced Monitoring implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class BankSpecializationSnapshot:
    """Snapshot of memory bank specialization at a specific timestep."""
    
    timestamp: float
    step: int
    cosine_similarities: torch.Tensor  # Pairwise similarities between banks
    orthogonality_loss: float
    bank_usage_stats: Dict[int, float]  # Usage percentage per bank
    bank_activation_patterns: Dict[int, float]  # Average activation per bank
    specialization_index: float  # Overall specialization measure
    diversity_score: float  # How diverse the banks are


@dataclass
class SpecializationAlert:
    """Alert for specialization issues."""
    
    timestamp: float
    step: int
    alert_type: str  # 'high_similarity', 'low_diversity', 'usage_imbalance'
    severity: str  # 'warning', 'critical'
    details: Dict[str, Any]
    message: str


class MemoryBankSpecializationMonitor:
    """
    Monitor memory bank specialization and diversity.
    
    Tracks how different memory banks specialize for different types of knowledge
    and ensures they maintain diversity to prevent collapse.
    """
    
    def __init__(self,
                 num_memory_banks: int = 4,
                 memory_size: int = 64,
                 memory_dim: int = 64,
                 history_size: int = 500,
                 similarity_warning_threshold: float = 0.8,
                 similarity_critical_threshold: float = 0.95,
                 usage_imbalance_threshold: float = 0.3,
                 diversity_threshold: float = 0.5):
        """
        Initialize memory bank specialization monitor.
        
        Args:
            num_memory_banks: Number of memory banks
            memory_size: Size of each memory bank
            memory_dim: Dimension of memory vectors
            history_size: Number of snapshots to keep
            similarity_warning_threshold: Cosine similarity warning threshold
            similarity_critical_threshold: Cosine similarity critical threshold
            usage_imbalance_threshold: Usage imbalance threshold
            diversity_threshold: Minimum diversity score threshold
        """
        self.num_memory_banks = num_memory_banks
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.history_size = history_size
        self.similarity_warning_threshold = similarity_warning_threshold
        self.similarity_critical_threshold = similarity_critical_threshold
        self.usage_imbalance_threshold = usage_imbalance_threshold
        self.diversity_threshold = diversity_threshold
        
        # History tracking
        self.specialization_history: deque = deque(maxlen=history_size)
        self.similarity_history: deque = deque(maxlen=history_size)
        self.usage_history: deque = deque(maxlen=history_size)
        
        # Usage tracking
        self.cumulative_usage = torch.zeros(num_memory_banks)
        self.usage_ema = torch.zeros(num_memory_banks)
        self.usage_momentum = 0.99
        
        # Alerts
        self.specialization_alerts: List[SpecializationAlert] = []
        self.last_alert_time = 0.0
        self.alert_cooldown = 30.0  # Seconds between alerts
        
        # Statistics
        self.total_steps = 0
        self.start_time = time.time()
        
        logger.info(f"MemoryBankSpecializationMonitor initialized for {num_memory_banks} banks")
    
    def update(self,
               memory_banks: torch.Tensor,
               router_weights: torch.Tensor,
               step: Optional[int] = None) -> BankSpecializationSnapshot:
        """
        Update specialization monitoring with current memory state.
        
        Args:
            memory_banks: Current memory banks [batch_size, num_banks, memory_size, memory_dim]
            router_weights: Router weights [batch_size, num_banks]
            step: Current training step
            
        Returns:
            BankSpecializationSnapshot with current statistics
        """
        current_time = time.time()
        if step is None:
            step = self.total_steps
        
        # Average across batch dimension
        if memory_banks.dim() == 4:
            avg_memory_banks = memory_banks.mean(dim=0)  # [num_banks, memory_size, memory_dim]
        else:
            avg_memory_banks = memory_banks
        
        if router_weights.dim() > 1:
            avg_router_weights = router_weights.mean(dim=0)  # [num_banks]
        else:
            avg_router_weights = router_weights
        
        # Calculate pairwise cosine similarities between banks
        cosine_similarities = self._calculate_bank_similarities(avg_memory_banks)
        
        # Calculate orthogonality loss
        orthogonality_loss = self._calculate_orthogonality_loss(avg_memory_banks)
        
        # Calculate usage statistics
        bank_usage_stats = self._calculate_usage_stats(avg_router_weights)
        
        # Calculate activation patterns
        bank_activation_patterns = self._calculate_activation_patterns(avg_memory_banks)
        
        # Calculate specialization index
        specialization_index = self._calculate_specialization_index(cosine_similarities)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(cosine_similarities, bank_usage_stats)
        
        # Create snapshot
        snapshot = BankSpecializationSnapshot(
            timestamp=current_time,
            step=step,
            cosine_similarities=cosine_similarities,
            orthogonality_loss=orthogonality_loss,
            bank_usage_stats=bank_usage_stats,
            bank_activation_patterns=bank_activation_patterns,
            specialization_index=specialization_index,
            diversity_score=diversity_score
        )
        
        # Update histories
        self.specialization_history.append(snapshot)
        self.similarity_history.append(cosine_similarities.clone())
        self.usage_history.append(torch.tensor(list(bank_usage_stats.values())))
        
        # Update usage tracking
        usage_tensor = torch.tensor(list(bank_usage_stats.values()))
        self.cumulative_usage += usage_tensor
        self.usage_ema = self.usage_momentum * self.usage_ema + (1 - self.usage_momentum) * usage_tensor
        
        # Check for specialization issues
        self._check_specialization_issues(snapshot, current_time, step)
        
        self.total_steps += 1
        
        return snapshot
    
    def _calculate_bank_similarities(self, memory_banks: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise cosine similarities between memory banks."""
        # Flatten each bank to a single vector
        flattened_banks = memory_banks.view(self.num_memory_banks, -1)  # [num_banks, memory_size * memory_dim]
        
        # Normalize for cosine similarity
        normalized_banks = F.normalize(flattened_banks, p=2, dim=1)
        
        # Calculate pairwise similarities
        similarities = torch.mm(normalized_banks, normalized_banks.t())
        
        return similarities
    
    def _calculate_orthogonality_loss(self, memory_banks: torch.Tensor) -> float:
        """Calculate orthogonality regularization loss."""
        # Flatten each bank
        flattened_banks = memory_banks.view(self.num_memory_banks, -1)
        
        # Normalize
        normalized_banks = F.normalize(flattened_banks, p=2, dim=1)
        
        # Calculate gram matrix
        gram_matrix = torch.mm(normalized_banks, normalized_banks.t())
        
        # Orthogonality loss: ||G - I||_F^2 where G is gram matrix, I is identity
        identity = torch.eye(self.num_memory_banks, device=gram_matrix.device)
        orthogonality_loss = torch.norm(gram_matrix - identity, p='fro').item() ** 2
        
        return orthogonality_loss
    
    def _calculate_usage_stats(self, router_weights: torch.Tensor) -> Dict[int, float]:
        """Calculate usage statistics for each memory bank."""
        usage_stats = {}
        
        for bank_id in range(self.num_memory_banks):
            usage_stats[bank_id] = router_weights[bank_id].item()
        
        return usage_stats
    
    def _calculate_activation_patterns(self, memory_banks: torch.Tensor) -> Dict[int, float]:
        """Calculate average activation patterns for each bank."""
        activation_patterns = {}
        
        for bank_id in range(self.num_memory_banks):
            bank_memory = memory_banks[bank_id]  # [memory_size, memory_dim]
            avg_activation = torch.mean(torch.abs(bank_memory)).item()
            activation_patterns[bank_id] = avg_activation
        
        return activation_patterns
    
    def _calculate_specialization_index(self, cosine_similarities: torch.Tensor) -> float:
        """Calculate overall specialization index (lower = more specialized)."""
        # Get off-diagonal elements (similarities between different banks)
        mask = ~torch.eye(self.num_memory_banks, dtype=torch.bool, device=cosine_similarities.device)
        off_diagonal_similarities = cosine_similarities[mask]
        
        # Specialization index is 1 - mean similarity (higher = more specialized)
        specialization_index = 1.0 - torch.mean(off_diagonal_similarities).item()
        
        return specialization_index
    
    def _calculate_diversity_score(self, cosine_similarities: torch.Tensor, usage_stats: Dict[int, float]) -> float:
        """Calculate diversity score combining similarity and usage balance."""
        # Similarity component (lower similarity = higher diversity)
        mask = ~torch.eye(self.num_memory_banks, dtype=torch.bool, device=cosine_similarities.device)
        avg_similarity = torch.mean(cosine_similarities[mask]).item()
        similarity_diversity = 1.0 - avg_similarity
        
        # Usage balance component (more balanced = higher diversity)
        usage_values = torch.tensor(list(usage_stats.values()))
        usage_entropy = -torch.sum(usage_values * torch.log(usage_values + 1e-8)).item()
        max_entropy = np.log(self.num_memory_banks)
        usage_diversity = usage_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Combined diversity score
        diversity_score = 0.7 * similarity_diversity + 0.3 * usage_diversity
        
        return diversity_score
    
    def _check_specialization_issues(self, snapshot: BankSpecializationSnapshot, 
                                   timestamp: float, step: int):
        """Check for specialization issues and generate alerts."""
        alerts_generated = []
        
        # Check for high similarity between banks
        max_similarity = torch.max(snapshot.cosine_similarities - torch.eye(self.num_memory_banks)).item()
        
        if max_similarity > self.similarity_critical_threshold:
            severity = 'critical'
            threshold = self.similarity_critical_threshold
        elif max_similarity > self.similarity_warning_threshold:
            severity = 'warning'
            threshold = self.similarity_warning_threshold
        else:
            severity = None
        
        if severity and timestamp - self.last_alert_time > self.alert_cooldown:
            alert = SpecializationAlert(
                timestamp=timestamp,
                step=step,
                alert_type='high_similarity',
                severity=severity,
                details={'max_similarity': max_similarity, 'threshold': threshold},
                message=f"High bank similarity detected: {max_similarity:.4f} > {threshold:.4f}"
            )
            alerts_generated.append(alert)
        
        # Check for usage imbalance
        usage_values = torch.tensor(list(snapshot.bank_usage_stats.values()))
        usage_std = torch.std(usage_values).item()
        usage_mean = torch.mean(usage_values).item()
        usage_cv = usage_std / (usage_mean + 1e-8)  # Coefficient of variation
        
        if usage_cv > self.usage_imbalance_threshold:
            if timestamp - self.last_alert_time > self.alert_cooldown:
                alert = SpecializationAlert(
                    timestamp=timestamp,
                    step=step,
                    alert_type='usage_imbalance',
                    severity='warning',
                    details={'usage_cv': usage_cv, 'threshold': self.usage_imbalance_threshold},
                    message=f"Usage imbalance detected: CV {usage_cv:.4f} > {self.usage_imbalance_threshold:.4f}"
                )
                alerts_generated.append(alert)
        
        # Check for low diversity
        if snapshot.diversity_score < self.diversity_threshold:
            if timestamp - self.last_alert_time > self.alert_cooldown:
                alert = SpecializationAlert(
                    timestamp=timestamp,
                    step=step,
                    alert_type='low_diversity',
                    severity='warning',
                    details={'diversity_score': snapshot.diversity_score, 'threshold': self.diversity_threshold},
                    message=f"Low diversity detected: {snapshot.diversity_score:.4f} < {self.diversity_threshold:.4f}"
                )
                alerts_generated.append(alert)
        
        # Add alerts and log
        for alert in alerts_generated:
            self.specialization_alerts.append(alert)
            if alert.severity == 'critical':
                logger.error(f"CRITICAL SPECIALIZATION ISSUE: {alert.message}")
            else:
                logger.warning(f"SPECIALIZATION WARNING: {alert.message}")
        
        if alerts_generated:
            self.last_alert_time = timestamp
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current specialization statistics."""
        if not self.specialization_history:
            return {}
        
        latest = self.specialization_history[-1]
        
        stats = {
            'specialization_index': latest.specialization_index,
            'diversity_score': latest.diversity_score,
            'orthogonality_loss': latest.orthogonality_loss,
            'bank_usage_stats': latest.bank_usage_stats,
            'bank_activation_patterns': latest.bank_activation_patterns,
            'total_alerts': len(self.specialization_alerts),
            'cumulative_usage': self.cumulative_usage.tolist(),
            'usage_ema': self.usage_ema.tolist()
        }
        
        # Add similarity statistics
        similarities = latest.cosine_similarities
        mask = ~torch.eye(self.num_memory_banks, dtype=torch.bool)
        off_diagonal = similarities[mask]
        
        stats.update({
            'max_similarity': torch.max(off_diagonal).item(),
            'min_similarity': torch.min(off_diagonal).item(),
            'avg_similarity': torch.mean(off_diagonal).item(),
            'similarity_std': torch.std(off_diagonal).item()
        })
        
        return stats
    
    def get_recent_alerts(self, max_alerts: int = 10) -> List[SpecializationAlert]:
        """Get recent specialization alerts."""
        return self.specialization_alerts[-max_alerts:] if self.specialization_alerts else []
    
    def is_specialization_healthy(self) -> Tuple[bool, str]:
        """Check if memory bank specialization is healthy."""
        if not self.specialization_history:
            return True, "No data available"
        
        latest = self.specialization_history[-1]
        
        # Check similarity
        similarities = latest.cosine_similarities
        mask = ~torch.eye(self.num_memory_banks, dtype=torch.bool)
        max_similarity = torch.max(similarities[mask]).item()
        
        if max_similarity > self.similarity_critical_threshold:
            return False, f"Critical: max similarity {max_similarity:.4f}"
        elif max_similarity > self.similarity_warning_threshold:
            return False, f"Warning: max similarity {max_similarity:.4f}"
        elif latest.diversity_score < self.diversity_threshold:
            return False, f"Warning: low diversity {latest.diversity_score:.4f}"
        else:
            return True, f"Healthy: diversity {latest.diversity_score:.4f}, max similarity {max_similarity:.4f}"
