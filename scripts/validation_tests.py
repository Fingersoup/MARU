#!/usr/bin/env python3
"""
Comprehensive Validation Tests for MARU Architecture

This script runs extensive tests to validate the MARU model's capabilities
before GitHub release, including:
1. Diverse prompt testing
2. Longer text generation
3. Memory capability validation
4. Architecture component testing
5. Performance benchmarking
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from maru import MARU, MARUConfig
from enhanced_mom_gru_config import get_conservative_config, get_baseline_config
from tokenizer import CharacterTokenizer

def load_model_and_tokenizer(checkpoint_path: str = "checkpoints/narrativeqa_epoch_1_final_converted.pt"):
    """Load the trained MARU model and tokenizer."""
    print(f"Loading model from {checkpoint_path}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print(f"Found config: {list(config_dict.keys())}")

        # Create MARU config - use original MoM-GRU for converted checkpoints
        use_enhanced = config_dict.get('use_enhanced_mom_gru', True)
        if 'converted' in checkpoint_path:
            use_enhanced = False  # Use original MoM-GRU for converted checkpoints
            print("Using original MoM-GRU for converted checkpoint")

        config = MARUConfig(
            vocab_size=config_dict.get('vocab_size', 1000),
            d_model=config_dict.get('d_model', 256),
            hidden_size=config_dict.get('hidden_size', 256),
            output_dim=config_dict.get('output_dim', 1),
            memory_size=config_dict.get('memory_size', 128),
            memory_dim=config_dict.get('memory_dim', 64),
            num_memories=config_dict.get('num_memories', 4),
            use_enhanced_mom_gru=use_enhanced,
            enhanced_mom_gru_config=get_baseline_config() if use_enhanced else None
        )

        vocab_size = config.vocab_size
        print(f"Model config: vocab_size={config.vocab_size}, d_model={config.d_model}, hidden_size={config.hidden_size}")

    else:
        print("No config found in checkpoint, using defaults")
        config = MARUConfig(vocab_size=1000, use_enhanced_mom_gru=True, enhanced_mom_gru_config=get_baseline_config())
        vocab_size = 1000

    # Create model
    model = MARU(config).to(device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
    else:
        print("No model_state_dict found in checkpoint")
        return None, None, None

    model.eval()

    # Load tokenizer (use the TRAINING tokenizer!)
    tokenizer = CharacterTokenizer()

    print(f"Model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt: str, max_length: int = 50, temperature: float = 0.8, top_k: int = 10):
    """Generate text from a prompt using the MARU model."""
    # Encode prompt using the training tokenizer
    input_tensor = tokenizer.encode(prompt, max_length=None, padding=False)
    input_ids = input_tensor.tolist()

    # Convert to tensor
    input_tensor = torch.tensor([input_ids], device=device)

    generated_ids = input_ids.copy()

    with torch.no_grad():
        # Get initial hidden state from the prompt
        if input_tensor.shape[1] > 1:
            # Process prompt through model
            output, (hidden, memory_state) = model(input_tensor, return_hidden=True)
            # Use the last hidden state
            current_hidden = hidden
            current_memory = memory_state
        else:
            current_hidden = None
            current_memory = None

        # Generate tokens one by one
        for _ in range(max_length):
            # Get last token
            if len(generated_ids) > 0:
                last_token = torch.tensor([[generated_ids[-1]]], device=device)
            else:
                # Start with first character if no prompt
                last_token = torch.tensor([[32]], device=device)  # Space character

            # Generate next token using generate_step
            if current_hidden is not None and hasattr(model, 'generate_step'):
                output, new_hidden, new_memory = model.generate_step(
                    last_token, current_hidden, current_memory
                )
                current_hidden = new_hidden
                current_memory = new_memory
            else:
                # First step or fallback
                output, (current_hidden, current_memory) = model(last_token, return_hidden=True)
                output = output[:, -1, :]  # Get last timestep

            # Apply temperature and sample
            logits = output / temperature
            probs = F.softmax(logits, dim=-1)

            # Sample next token (limit to actual vocab size)
            vocab_size = min(tokenizer.vocab_size, logits.shape[-1])
            probs_truncated = probs[:vocab_size]
            probs_truncated = probs_truncated / probs_truncated.sum()  # Renormalize
            next_token = torch.multinomial(probs_truncated, 1).item()

            # Check for reasonable stopping
            if next_token >= tokenizer.vocab_size:
                break

            generated_ids.append(next_token)

    # Decode final result using training tokenizer
    try:
        final_tensor = torch.tensor(generated_ids)
        generated_text = tokenizer.decode(final_tensor, skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Generated {len(generated_ids)} tokens but couldn't decode: {e}"

def test_diverse_prompts(model, tokenizer, device):
    """Test the model with diverse types of prompts."""
    print("\n" + "="*60)
    print("DIVERSE PROMPT TESTING")
    print("="*60)
    
    test_prompts = [
        # Basic words
        "Hello",
        "The",
        "AI",
        "Python",
        
        # Phrases
        "Once upon",
        "In the beginning",
        "The quick brown",
        "Machine learning",
        
        # Questions
        "What is",
        "How do",
        "Why does",
        
        # Technical terms
        "Neural network",
        "Algorithm",
        "Computer",
        "Data science",
        
        # Creative prompts
        "The dragon",
        "In a world where",
        "The scientist discovered",
        "Magic and technology"
    ]
    
    results = {}
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            generated = generate_text(model, tokenizer, device, prompt, max_length=30, temperature=0.8)
            print(f"Output: '{generated}'")
            results[prompt] = generated
        except Exception as e:
            print(f"Error: {e}")
            results[prompt] = f"ERROR: {e}"
    
    return results

def test_longer_generation(model, tokenizer, device):
    """Test longer text generation capabilities."""
    print("\n" + "="*60)
    print("LONGER TEXT GENERATION TESTING")
    print("="*60)
    
    prompts = [
        "Once upon a time",
        "The future of AI",
        "In the year 2050"
    ]
    
    results = {}
    
    for prompt in prompts:
        print(f"\nLong generation for: '{prompt}'")
        try:
            # Generate longer text with different temperatures
            for temp in [0.5, 0.8, 1.0]:
                generated = generate_text(model, tokenizer, device, prompt, max_length=100, temperature=temp)
                print(f"Temperature {temp}: '{generated}'")
                results[f"{prompt}_temp_{temp}"] = generated
        except Exception as e:
            print(f"Error: {e}")
            results[prompt] = f"ERROR: {e}"
    
    return results

def test_memory_capabilities(model, tokenizer, device):
    """Test the model's memory capabilities with repeated patterns."""
    print("\n" + "="*60)
    print("MEMORY CAPABILITY TESTING")
    print("="*60)
    
    # Test with repeated patterns to see if memory helps
    memory_prompts = [
        "A B A B A B",  # Pattern completion
        "1 2 3 1 2 3",  # Numeric pattern
        "red blue red blue",  # Color pattern
        "cat dog cat dog"  # Word pattern
    ]
    
    results = {}
    
    for prompt in memory_prompts:
        print(f"\nMemory test: '{prompt}'")
        try:
            generated = generate_text(model, tokenizer, device, prompt, max_length=40, temperature=0.6)
            print(f"Output: '{generated}'")
            results[prompt] = generated
        except Exception as e:
            print(f"Error: {e}")
            results[prompt] = f"ERROR: {e}"
    
    return results

def test_architecture_components(model, device):
    """Test individual architecture components."""
    print("\n" + "="*60)
    print("ARCHITECTURE COMPONENT TESTING")
    print("="*60)
    
    results = {}
    
    # Test model structure
    print("Model Architecture:")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"- Embedding dimension: {model.embedding.embedding_dim}")
    print(f"- Model dimension: {model.config.d_model}")
    print(f"- Hidden size: {model.config.hidden_size}")
    print(f"- Memory size: {model.config.memory_size}")
    print(f"- Number of memories: {model.config.num_memories}")
    
    # Test forward pass with dummy input
    print("\nTesting forward pass...")
    try:
        dummy_input = torch.randint(0, 256, (1, 10), device=device)
        with torch.no_grad():
            output, (hidden, memory_state) = model(dummy_input, return_hidden=True)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  Memory state type: {type(memory_state)}")
        results['forward_pass'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        results['forward_pass'] = f'FAILED: {e}'

    # Test single step inference
    print("\nTesting single step inference...")
    try:
        single_input = torch.randint(0, 256, (1, 1), device=device)
        with torch.no_grad():
            if hasattr(model, 'generate_step'):
                output, hidden, memory = model.generate_step(single_input, None, None)
                print(f"✓ Single step inference successful (generate_step)")
            else:
                output, (hidden, memory) = model(single_input, return_hidden=True)
                output = output[:, -1, :]  # Get last timestep
                print(f"✓ Single step inference successful (fallback)")
        print(f"  Input shape: {single_input.shape}")
        print(f"  Output shape: {output.shape}")
        results['single_step'] = 'SUCCESS'
    except Exception as e:
        print(f"✗ Single step inference failed: {e}")
        results['single_step'] = f'FAILED: {e}'
    
    return results

def benchmark_performance(model, tokenizer, device):
    """Benchmark model performance."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    results = {}
    
    # Test inference speed
    print("Testing inference speed...")
    prompt = "Hello world"
    input_tensor = tokenizer.encode(prompt, max_length=None, padding=False)
    tokens = input_tensor.tolist()
    input_ids = torch.tensor([tokens], device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)
    
    # Benchmark
    num_runs = 50
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Tokens per second: {len(tokens) / avg_time:.2f}")
    
    results['avg_inference_time_ms'] = avg_time * 1000
    results['tokens_per_second'] = len(tokens) / avg_time
    
    # Test memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(input_ids)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")
        results['peak_memory_mb'] = peak_memory
    
    return results

def run_all_tests():
    """Run all validation tests."""
    print("MARU Model Validation Tests")
    print("="*60)
    
    # Load model
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    all_results = {}
    
    # Run tests
    try:
        all_results['diverse_prompts'] = test_diverse_prompts(model, tokenizer, device)
        all_results['longer_generation'] = test_longer_generation(model, tokenizer, device)
        all_results['memory_capabilities'] = test_memory_capabilities(model, tokenizer, device)
        all_results['architecture_components'] = test_architecture_components(model, device)
        all_results['performance_benchmark'] = benchmark_performance(model, tokenizer, device)
        
        # Save results
        with open('validation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print("Results saved to validation_results.json")
        print("\nSummary:")
        print(f"✓ Diverse prompts tested: {len(all_results['diverse_prompts'])}")
        print(f"✓ Longer generation tested: {len(all_results['longer_generation'])}")
        print(f"✓ Memory capabilities tested: {len(all_results['memory_capabilities'])}")
        print(f"✓ Architecture components: {all_results['architecture_components'].get('forward_pass', 'Unknown')}")
        print(f"✓ Performance benchmarked: {all_results['performance_benchmark'].get('avg_inference_time_ms', 'Unknown'):.2f} ms avg")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
