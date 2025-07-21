"""
Teacher Model Interface for Knowledge Distillation

This module provides a unified interface for teacher models used in knowledge distillation
with the MARU architecture. It supports both local and HuggingFace models with configurable
temperature scaling, top-k/top-p sampling, and vocabulary mapping.

Key Features:
- Unified interface for different model types (causal LM, masked LM)
- Temperature scaling and sampling controls
- Vocabulary alignment with student models
- Memory-efficient batch processing
- Support for both local checkpoints and HuggingFace models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM,
        GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer
    )
    HF_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers")
    HF_AVAILABLE = False
    # Define dummy classes for type hints
    PreTrainedModel = object
    PreTrainedTokenizer = object


class TeacherModel:
    """
    Unified interface for teacher models in knowledge distillation.
    
    Supports both HuggingFace models and local checkpoints with configurable
    temperature scaling, sampling controls, and vocabulary mapping.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        temperature: float = 3.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        model_type: str = "causal",
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the teacher model.
        
        Args:
            model_name: Name/path of the model (HuggingFace model ID or local path)
            device: Device to use ('auto', 'cuda', 'cpu')
            temperature: Temperature for softmax scaling (default: 3.0)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            model_type: Type of model ('causal', 'masked')
            torch_dtype: PyTorch dtype for model weights
            trust_remote_code: Whether to trust remote code for custom models
            cache_dir: Directory to cache downloaded models
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers library required for TeacherModel")
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set dtype
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._vocab_mapping = None
        
        self._load_model()
        
        logger.info(f"Teacher model loaded: {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model based on type
            if self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    device_map="auto" if self.device == "cuda" else None
                )
            elif self.model_type == "masked":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Move to device if not using device_map
            if not (self.device == "cuda" and hasattr(self.model, 'hf_device_map')):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Resize token embeddings if tokenizer was modified
            if len(self.tokenizer) != self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
        except Exception as e:
            logger.error(f"Failed to load teacher model {self.model_name}: {e}")
            raise
    
    def extract_logits(self, text_batch: List[str]) -> torch.Tensor:
        """
        Extract logits from a batch of text inputs.
        
        Args:
            text_batch: List of text strings to process
            
        Returns:
            Tensor of shape (batch_size, vocab_size) containing logits
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not properly initialized")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Reasonable default
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if self.model_type == "causal":
                outputs = self.model(**inputs)
                # For causal models, take the last token's logits
                logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            elif self.model_type == "masked":
                outputs = self.model(**inputs)
                # For masked models, average over sequence length
                logits = outputs.logits.mean(dim=1)  # (batch_size, vocab_size)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return logits
    
    def extract_logits_with_temperature(self, text_batch: List[str]) -> torch.Tensor:
        """
        Extract temperature-scaled logits from a batch of text inputs.
        
        Args:
            text_batch: List of text strings to process
            
        Returns:
            Temperature-scaled logits tensor
        """
        logits = self.extract_logits(text_batch)
        return logits / self.temperature
    
    def get_vocab_mapping(self) -> Dict[str, int]:
        """
        Get vocabulary mapping for the teacher model.
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        if self._vocab_mapping is None:
            if hasattr(self.tokenizer, 'vocab'):
                self._vocab_mapping = dict(self.tokenizer.vocab)
            else:
                # For tokenizers without direct vocab access
                self._vocab_mapping = {
                    self.tokenizer.decode([i]): i 
                    for i in range(self.tokenizer.vocab_size)
                }
        
        return self._vocab_mapping
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the teacher model."""
        return len(self.tokenizer) if self.tokenizer else 0
    
    def generate_with_sampling(
        self,
        text_batch: List[str],
        max_new_tokens: int = 50,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate text with configured sampling parameters.
        
        Args:
            text_batch: List of input text strings
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            List of generated text strings
        """
        if self.model_type != "causal":
            raise ValueError("Text generation only supported for causal models")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": self.temperature if do_sample else None,
            "top_k": self.top_k if do_sample else None,
            "top_p": self.top_p if do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def compute_perplexity(self, text_batch: List[str]) -> List[float]:
        """
        Compute perplexity for a batch of text inputs.
        
        Args:
            text_batch: List of text strings
            
        Returns:
            List of perplexity values
        """
        if self.model_type != "causal":
            raise ValueError("Perplexity computation only supported for causal models")
        
        perplexities = []
        
        for text in text_batch:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return perplexities
    
    def align_vocabulary_with_student(self, student_vocab: Dict[str, int]) -> Dict[int, int]:
        """
        Create mapping between teacher and student vocabularies.
        
        Args:
            student_vocab: Student model vocabulary mapping
            
        Returns:
            Dictionary mapping teacher token IDs to student token IDs
        """
        teacher_vocab = self.get_vocab_mapping()
        alignment = {}
        
        # Create reverse mapping for student vocab
        student_id_to_token = {v: k for k, v in student_vocab.items()}
        
        for teacher_token, teacher_id in teacher_vocab.items():
            if teacher_token in student_vocab:
                # Direct match
                alignment[teacher_id] = student_vocab[teacher_token]
            else:
                # Try to find closest match or use UNK token
                if '<unk>' in student_vocab:
                    alignment[teacher_id] = student_vocab['<unk>']
                elif '[UNK]' in student_vocab:
                    alignment[teacher_id] = student_vocab['[UNK]']
                else:
                    # Use first token as fallback
                    alignment[teacher_id] = 0
        
        return alignment
    
    def save_config(self, path: str):
        """Save teacher model configuration."""
        config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "model_type": self.model_type,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, config_path: str) -> 'TeacherModel':
        """Load teacher model from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Convert torch_dtype string back to dtype
        if 'torch_dtype' in config:
            dtype_str = config.pop('torch_dtype')
            if dtype_str == "torch.float16":
                config['torch_dtype'] = torch.float16
            elif dtype_str == "torch.float32":
                config['torch_dtype'] = torch.float32
        
        return cls(**config)
    
    def __repr__(self) -> str:
        return (f"TeacherModel(model_name='{self.model_name}', "
                f"device='{self.device}', temperature={self.temperature})")


def create_teacher_model(
    model_name: str,
    device: str = "auto",
    temperature: float = 3.0,
    **kwargs
) -> TeacherModel:
    """
    Factory function to create a teacher model with default settings.
    
    Args:
        model_name: Name/path of the model
        device: Device to use
        temperature: Temperature for scaling
        **kwargs: Additional arguments for TeacherModel
        
    Returns:
        Configured TeacherModel instance
    """
    return TeacherModel(
        model_name=model_name,
        device=device,
        temperature=temperature,
        **kwargs
    )
