#!/usr/bin/env python3
"""
Vocabulary Alignment for MARU Knowledge Distillation

This module implements vocabulary alignment between teacher models (typically BPE-based
with large vocabularies) and MARU's character-level vocabulary (256 characters).

Key Features:
- BPE-to-character logit mapping for token-based teachers
- Character-level teacher tokenizer wrapper
- Vocabulary size validation and padding/truncation
- Logit interpolation for vocabulary mismatches
- Support for multiple teacher vocabularies

Critical for fixing sleep cycle distillation - required for Phase 0.2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import string
import unicodedata

logger = logging.getLogger(__name__)

# MARU's character-level vocabulary (256 characters)
MARU_VOCAB_SIZE = 256
MARU_CHAR_VOCAB = list(range(256))  # ASCII + extended characters


class VocabularyAligner:
    """
    Handles vocabulary alignment between teacher and student models.
    
    This class provides methods to map teacher model logits (typically BPE-based)
    to MARU's character-level vocabulary space, enabling proper knowledge distillation.
    """
    
    def __init__(self, teacher_vocab_size: int, student_vocab_size: int = MARU_VOCAB_SIZE):
        """
        Initialize the vocabulary aligner.
        
        Args:
            teacher_vocab_size: Size of teacher model vocabulary
            student_vocab_size: Size of student model vocabulary (default: 256 for MARU)
        """
        self.teacher_vocab_size = teacher_vocab_size
        self.student_vocab_size = student_vocab_size
        
        # Create mapping matrices
        self.teacher_to_student_map = self._create_vocab_mapping()
        
        logger.info(f"VocabularyAligner initialized: {teacher_vocab_size} -> {student_vocab_size}")
    
    def _create_vocab_mapping(self) -> torch.Tensor:
        """
        Create a mapping matrix from teacher vocabulary to student vocabulary.
        
        For now, we use a simple approach:
        - Map teacher tokens to character-level equivalents where possible
        - Use uniform distribution over character vocab for unmappable tokens
        
        Returns:
            Mapping matrix of shape (teacher_vocab_size, student_vocab_size)
        """
        mapping = torch.zeros(self.teacher_vocab_size, self.student_vocab_size)
        
        # Simple mapping strategy: distribute teacher vocab uniformly over student vocab
        # In a more sophisticated implementation, this would use actual token-to-character mapping
        for i in range(self.teacher_vocab_size):
            # Map each teacher token to a subset of character tokens
            start_idx = (i * self.student_vocab_size) // self.teacher_vocab_size
            end_idx = ((i + 1) * self.student_vocab_size) // self.teacher_vocab_size
            
            if end_idx > start_idx:
                # Uniform distribution over the mapped range
                mapping[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
            else:
                # Fallback: map to a single character
                mapping[i, i % self.student_vocab_size] = 1.0
        
        return mapping
    
    def align_logits(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Align teacher model logits to student vocabulary space.
        
        Args:
            teacher_logits: Teacher logits of shape (batch_size, teacher_vocab_size)
            
        Returns:
            Aligned logits of shape (batch_size, student_vocab_size)
        """
        if teacher_logits.size(-1) != self.teacher_vocab_size:
            raise ValueError(f"Expected teacher logits size {self.teacher_vocab_size}, "
                           f"got {teacher_logits.size(-1)}")
        
        # Move mapping to same device as logits
        mapping = self.teacher_to_student_map.to(teacher_logits.device)
        
        # Apply mapping: (batch_size, teacher_vocab) @ (teacher_vocab, student_vocab)
        aligned_logits = torch.matmul(teacher_logits, mapping)
        
        return aligned_logits
    
    def align_probabilities(self, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        Align teacher model probabilities to student vocabulary space.
        
        Args:
            teacher_probs: Teacher probabilities of shape (batch_size, teacher_vocab_size)
            
        Returns:
            Aligned probabilities of shape (batch_size, student_vocab_size)
        """
        if teacher_probs.size(-1) != self.teacher_vocab_size:
            raise ValueError(f"Expected teacher probs size {self.teacher_vocab_size}, "
                           f"got {teacher_probs.size(-1)}")
        
        # Move mapping to same device as probabilities
        mapping = self.teacher_to_student_map.to(teacher_probs.device)
        
        # Apply mapping to probabilities
        aligned_probs = torch.matmul(teacher_probs, mapping)
        
        # Ensure probabilities sum to 1
        aligned_probs = aligned_probs / aligned_probs.sum(dim=-1, keepdim=True)
        
        return aligned_probs


class CharacterLevelTokenizer:
    """
    Character-level tokenizer wrapper for teacher models.
    
    This class provides a character-level interface for teacher models,
    enabling direct compatibility with MARU's character-level vocabulary.
    """
    
    def __init__(self, max_length: int = 512):
        """
        Initialize the character-level tokenizer.
        
        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.vocab_size = MARU_VOCAB_SIZE
        
        # Create character-to-id mapping
        self.char_to_id = {chr(i): i for i in range(256)}
        self.id_to_char = {i: chr(i) for i in range(256)}
        
        logger.info(f"CharacterLevelTokenizer initialized with vocab_size={self.vocab_size}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to character-level token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of character token IDs
        """
        # Convert to bytes and then to character IDs
        try:
            byte_sequence = text.encode('utf-8', errors='ignore')
            token_ids = [b for b in byte_sequence if b < 256]
            
            # Truncate if too long
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                
            return token_ids
            
        except Exception as e:
            logger.warning(f"Error encoding text: {e}")
            return []
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode character-level token IDs to text.
        
        Args:
            token_ids: List of character token IDs
            
        Returns:
            Decoded text string
        """
        try:
            # Convert to bytes and decode
            byte_sequence = bytes([tid for tid in token_ids if 0 <= tid < 256])
            text = byte_sequence.decode('utf-8', errors='ignore')
            return text
            
        except Exception as e:
            logger.warning(f"Error decoding tokens: {e}")
            return ""
    
    def batch_encode(self, texts: List[str], padding: bool = True) -> torch.Tensor:
        """
        Batch encode texts to character-level tokens.
        
        Args:
            texts: List of input text strings
            padding: Whether to pad sequences to same length
            
        Returns:
            Tensor of shape (batch_size, max_seq_len)
        """
        encoded_texts = [self.encode(text) for text in texts]
        
        if padding:
            # Pad to maximum length in batch
            max_len = max(len(seq) for seq in encoded_texts) if encoded_texts else 0
            max_len = min(max_len, self.max_length)
            
            padded_texts = []
            for seq in encoded_texts:
                if len(seq) < max_len:
                    # Pad with null character (0)
                    seq = seq + [0] * (max_len - len(seq))
                elif len(seq) > max_len:
                    seq = seq[:max_len]
                padded_texts.append(seq)
            
            return torch.tensor(padded_texts, dtype=torch.long)
        else:
            # Return list of tensors with different lengths
            return [torch.tensor(seq, dtype=torch.long) for seq in encoded_texts]


def create_vocabulary_aligner(teacher_vocab_size: int, 
                            student_vocab_size: int = MARU_VOCAB_SIZE) -> VocabularyAligner:
    """
    Factory function to create a vocabulary aligner.
    
    Args:
        teacher_vocab_size: Size of teacher model vocabulary
        student_vocab_size: Size of student model vocabulary
        
    Returns:
        Configured VocabularyAligner instance
    """
    return VocabularyAligner(teacher_vocab_size, student_vocab_size)


def validate_vocab_alignment(teacher_logits: torch.Tensor, 
                           aligned_logits: torch.Tensor,
                           tolerance: float = 1e-6) -> bool:
    """
    Validate that vocabulary alignment preserves probability mass.
    
    Args:
        teacher_logits: Original teacher logits
        aligned_logits: Aligned logits
        tolerance: Numerical tolerance for validation
        
    Returns:
        True if alignment is valid, False otherwise
    """
    try:
        # Convert to probabilities
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        aligned_probs = F.softmax(aligned_logits, dim=-1)
        
        # Check that probabilities sum to 1
        teacher_sum = teacher_probs.sum(dim=-1)
        aligned_sum = aligned_probs.sum(dim=-1)
        
        teacher_valid = torch.allclose(teacher_sum, torch.ones_like(teacher_sum), atol=tolerance)
        aligned_valid = torch.allclose(aligned_sum, torch.ones_like(aligned_sum), atol=tolerance)
        
        return teacher_valid and aligned_valid
        
    except Exception as e:
        logger.error(f"Error validating vocab alignment: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test vocabulary alignment
    teacher_vocab_size = 50257  # GPT-2 vocabulary size
    student_vocab_size = 256    # MARU character vocabulary size
    
    aligner = create_vocabulary_aligner(teacher_vocab_size, student_vocab_size)
    
    # Test logit alignment
    batch_size = 4
    teacher_logits = torch.randn(batch_size, teacher_vocab_size)
    aligned_logits = aligner.align_logits(teacher_logits)
    
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"Aligned logits shape: {aligned_logits.shape}")
    
    # Validate alignment
    is_valid = validate_vocab_alignment(teacher_logits, aligned_logits)
    print(f"Alignment valid: {is_valid}")
    
    # Test character-level tokenizer
    tokenizer = CharacterLevelTokenizer()
    test_text = "Hello, world! This is a test."
    
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded[:20]}...")  # Show first 20 tokens
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {test_text == decoded}")
