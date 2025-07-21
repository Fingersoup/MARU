"""
Simple character-level tokenizer for the MARU project.

This tokenizer converts text sequences into integer tensors and vice versa,
using character-level encoding.
"""

import torch
from typing import List, Dict, Union


class CharacterTokenizer:
    """
    A simple character-level tokenizer that encodes and decodes text sequences
    into integer tensors.
    """
    
    def __init__(self, vocabulary: Union[str, List[str]] = None):
        """
        Initialize the tokenizer with a vocabulary.
        
        Args:
            vocabulary: Either a string containing all characters to include,
                       or a list of characters. If None, uses a default ASCII set.
        """
        if vocabulary is None:
            # Default vocabulary: printable ASCII characters
            self.vocabulary = self._create_default_vocabulary()
        elif isinstance(vocabulary, str):
            # Convert string to list of unique characters
            self.vocabulary = list(set(vocabulary))
        else:
            # Assume it's already a list
            self.vocabulary = list(set(vocabulary))
        
        # Sort for consistency
        self.vocabulary.sort()
        
        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
        # Insert special tokens at the beginning
        if self.pad_token not in self.vocabulary:
            self.vocabulary.insert(0, self.pad_token)
        if self.unk_token not in self.vocabulary:
            self.vocabulary.insert(1, self.unk_token)
        
        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocabulary)}
        
        self.vocab_size = len(self.vocabulary)
        self.pad_id = self.char_to_id[self.pad_token]
        self.unk_id = self.char_to_id[self.unk_token]
    
    def _create_default_vocabulary(self) -> List[str]:
        """Create a default vocabulary with printable ASCII characters."""
        import string
        # Include letters, digits, punctuation, and space
        chars = string.ascii_letters + string.digits + string.punctuation + ' '
        return list(set(chars))
    
    def encode(self, text: str, max_length: int = None, padding: bool = True) -> torch.Tensor:
        """
        Encode text into a tensor of token IDs.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length. If None, no truncation/padding
            padding: Whether to pad sequences to max_length
            
        Returns:
            torch.Tensor: Tensor of token IDs with shape (sequence_length,)
        """
        # Convert characters to IDs
        token_ids = []
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                token_ids.append(self.unk_id)
        
        # Handle max_length
        if max_length is not None:
            if len(token_ids) > max_length:
                # Truncate
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length and padding:
                # Pad
                token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode a tensor of token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens like PAD and UNK
            
        Returns:
            str: Decoded text
        """
        if token_ids.dim() > 1:
            raise ValueError("Input tensor must be 1-dimensional")
        
        chars = []
        for token_id in token_ids.tolist():
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                if skip_special_tokens and char in [self.pad_token, self.unk_token]:
                    continue
                chars.append(char)
            else:
                if not skip_special_tokens:
                    chars.append(self.unk_token)
        
        return ''.join(chars)
    
    def encode_batch(self, texts: List[str], max_length: int = None, padding: bool = True) -> torch.Tensor:
        """
        Encode a batch of texts into a tensor.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            
        Returns:
            torch.Tensor: Tensor with shape (batch_size, sequence_length)
        """
        if max_length is None and padding:
            # Find the maximum length in the batch
            max_length = max(len(text) for text in texts)
        
        encoded_texts = []
        for text in texts:
            encoded = self.encode(text, max_length=max_length, padding=padding)
            encoded_texts.append(encoded)
        
        return torch.stack(encoded_texts)
    
    def decode_batch(self, token_ids_batch: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID tensors back to texts.
        
        Args:
            token_ids_batch: Tensor with shape (batch_size, sequence_length)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List[str]: List of decoded texts
        """
        if token_ids_batch.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional (batch_size, sequence_length)")
        
        decoded_texts = []
        for token_ids in token_ids_batch:
            decoded = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            decoded_texts.append(decoded)
        
        return decoded_texts
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def get_vocabulary(self) -> List[str]:
        """Return the vocabulary as a list."""
        return self.vocabulary.copy()
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        return f"CharacterTokenizer(vocab_size={self.vocab_size})"
