#!/usr/bin/env python3
"""
Teacher Model Evaluation Script

This script evaluates different teacher model options for knowledge distillation:
1. Character-level models (DistilGPT-2, small GPT-2 with char tokenizer)
2. Small token-level models (DialoGPT-medium, GPT-2-small)
3. Benchmarks inference speed (target >500 tokens/sec)
4. Measures VRAM requirements to ensure compatibility with MARU training

Usage:
    python scripts/evaluate_teacher_models.py --models all --benchmark-speed --measure-vram
"""

import argparse
import time
import json
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        GPT2LMHeadModel, GPT2Tokenizer,
        DistilBertTokenizer, DistilBertForMaskedLM
    )
    HF_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers")
    HF_AVAILABLE = False


class TeacherModelEvaluator:
    """Evaluates teacher models for knowledge distillation compatibility."""
    
    def __init__(self, device: str = "auto", max_memory_gb: float = 8.0):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            max_memory_gb: Maximum VRAM to use for teacher models
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.max_memory_gb = max_memory_gb
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {
            "ram_gb": psutil.virtual_memory().used / 1e9,
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            memory_info.update({
                "vram_gb": torch.cuda.memory_allocated() / 1e9,
                "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "vram_max_gb": torch.cuda.max_memory_allocated() / 1e9
            })
        
        return memory_info
    
    def benchmark_inference_speed(self, model, tokenizer, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed for a model.
        
        Args:
            model: The model to benchmark
            tokenizer: The tokenizer for the model
            num_samples: Number of inference samples to run
            
        Returns:
            Dictionary with speed metrics
        """
        logger.info(f"Benchmarking inference speed with {num_samples} samples...")
        
        # Prepare test inputs
        test_prompts = [
            "The quick brown fox",
            "In a distant galaxy",
            "Machine learning is",
            "The future of AI",
            "Once upon a time"
        ]
        
        # Tokenize inputs
        inputs = []
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            inputs.append(tokens)
        
        model.eval()
        total_tokens = 0
        total_time = 0
        
        with torch.no_grad():
            for i in range(num_samples):
                input_tokens = inputs[i % len(inputs)]
                
                start_time = time.time()
                
                # Generate tokens
                outputs = model.generate(
                    input_tokens,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                end_time = time.time()
                
                # Count generated tokens
                generated_tokens = outputs.shape[1] - input_tokens.shape[1]
                total_tokens += generated_tokens
                total_time += (end_time - start_time)
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "total_tokens": total_tokens,
            "total_time": total_time,
            "tokens_per_second": tokens_per_second,
            "avg_time_per_token": total_time / total_tokens if total_tokens > 0 else 0
        }
    
    def evaluate_model(self, model_name: str, model_type: str = "causal") -> Dict[str, Any]:
        """
        Evaluate a single teacher model.
        
        Args:
            model_name: Name/path of the model
            model_type: Type of model ('causal', 'masked')
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating model: {model_name}")
        
        if not HF_AVAILABLE:
            return {"error": "Transformers library not available"}
        
        try:
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Record initial memory
            initial_memory = self.get_memory_usage()
            
            # Load model and tokenizer
            start_time = time.time()
            
            if model_type == "causal":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                # Set pad token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
            else:  # masked language model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            if self.device != "cuda" or "device_map" not in locals():
                model = model.to(self.device)
            
            load_time = time.time() - start_time
            
            # Record memory after loading
            post_load_memory = self.get_memory_usage()
            
            # Get model info
            num_parameters = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Benchmark speed (only for causal models)
            speed_metrics = {}
            if model_type == "causal":
                try:
                    speed_metrics = self.benchmark_inference_speed(model, tokenizer, num_samples=20)
                except Exception as e:
                    logger.warning(f"Speed benchmark failed: {e}")
                    speed_metrics = {"error": str(e)}
            
            # Test vocabulary mapping
            vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
            
            # Calculate memory usage
            memory_usage = {
                "vram_used_gb": (post_load_memory.get("vram_gb", 0) - 
                               initial_memory.get("vram_gb", 0)),
                "ram_used_gb": (post_load_memory.get("ram_gb", 0) - 
                              initial_memory.get("ram_gb", 0)),
                "peak_vram_gb": post_load_memory.get("vram_max_gb", 0)
            }
            
            # Cleanup
            del model, tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "model_name": model_name,
                "model_type": model_type,
                "success": True,
                "load_time": load_time,
                "num_parameters": num_parameters,
                "num_trainable_parameters": num_trainable,
                "vocab_size": vocab_size,
                "memory_usage": memory_usage,
                "speed_metrics": speed_metrics,
                "compatible_with_maru": memory_usage["vram_used_gb"] < self.max_memory_gb,
                "meets_speed_target": speed_metrics.get("tokens_per_second", 0) > 500
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {
                "model_name": model_name,
                "model_type": model_type,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all predefined teacher model candidates."""
        
        # Define model candidates
        model_candidates = {
            # Character-level models
            "distilgpt2": {"name": "distilgpt2", "type": "causal"},
            "gpt2": {"name": "gpt2", "type": "causal"},
            
            # Small token-level models  
            "dialogpt-medium": {"name": "microsoft/DialoGPT-medium", "type": "causal"},
            "dialogpt-small": {"name": "microsoft/DialoGPT-small", "type": "causal"},
            
            # Additional small models
            "gpt2-medium": {"name": "gpt2-medium", "type": "causal"},
            "distilbert": {"name": "distilbert-base-uncased", "type": "masked"}
        }
        
        results = {}
        
        for model_id, config in model_candidates.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {model_id}")
            logger.info(f"{'='*50}")
            
            result = self.evaluate_model(config["name"], config["type"])
            results[model_id] = result
            
            # Print summary
            if result["success"]:
                logger.info(f"✅ {model_id}: {result['num_parameters']:,} params, "
                          f"{result['memory_usage']['vram_used_gb']:.1f}GB VRAM")
                if "tokens_per_second" in result["speed_metrics"]:
                    logger.info(f"   Speed: {result['speed_metrics']['tokens_per_second']:.1f} tokens/sec")
            else:
                logger.error(f"❌ {model_id}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        
        report = []
        report.append("# Teacher Model Evaluation Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append(f"Max Memory Budget: {self.max_memory_gb} GB")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("| Model | Parameters | VRAM (GB) | Speed (tok/s) | Compatible | Meets Speed |")
        report.append("|-------|------------|-----------|---------------|------------|-------------|")
        
        for model_id, result in results.items():
            if result["success"]:
                params = f"{result['num_parameters']:,}"
                vram = f"{result['memory_usage']['vram_used_gb']:.1f}"
                speed = result['speed_metrics'].get('tokens_per_second', 0)
                speed_str = f"{speed:.1f}" if speed > 0 else "N/A"
                compatible = "✅" if result["compatible_with_maru"] else "❌"
                meets_speed = "✅" if result["meets_speed_target"] else "❌"
                
                report.append(f"| {model_id} | {params} | {vram} | {speed_str} | {compatible} | {meets_speed} |")
            else:
                report.append(f"| {model_id} | ERROR | ERROR | ERROR | ❌ | ❌ |")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        compatible_models = [
            (model_id, result) for model_id, result in results.items()
            if result["success"] and result["compatible_with_maru"]
        ]
        
        if compatible_models:
            # Sort by speed
            compatible_models.sort(
                key=lambda x: x[1]['speed_metrics'].get('tokens_per_second', 0),
                reverse=True
            )
            
            report.append("### Compatible Models (sorted by speed):")
            for model_id, result in compatible_models:
                speed = result['speed_metrics'].get('tokens_per_second', 0)
                report.append(f"- **{model_id}**: {speed:.1f} tokens/sec, "
                            f"{result['memory_usage']['vram_used_gb']:.1f}GB VRAM")
        
        # Best recommendations
        fast_models = [
            (model_id, result) for model_id, result in compatible_models
            if result["meets_speed_target"]
        ]
        
        if fast_models:
            best_model = fast_models[0]
            report.append(f"\n### Recommended Teacher Model: **{best_model[0]}**")
            report.append(f"- Speed: {best_model[1]['speed_metrics']['tokens_per_second']:.1f} tokens/sec")
            report.append(f"- VRAM: {best_model[1]['memory_usage']['vram_used_gb']:.1f} GB")
            report.append(f"- Parameters: {best_model[1]['num_parameters']:,}")
        
        return "\n".join(report)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate teacher models for MARU knowledge distillation")
    parser.add_argument("--models", choices=["all", "small", "character"], default="all",
                       help="Which models to evaluate")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use for evaluation")
    parser.add_argument("--max-memory", type=float, default=8.0,
                       help="Maximum VRAM budget in GB")
    parser.add_argument("--output", type=str, default="teacher_model_evaluation.json",
                       help="Output file for results")
    parser.add_argument("--report", type=str, default="teacher_model_report.md",
                       help="Output file for markdown report")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TeacherModelEvaluator(device=args.device, max_memory_gb=args.max_memory)
    
    # Run evaluation
    logger.info("Starting teacher model evaluation...")
    results = evaluator.evaluate_all_models()
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")
    
    # Generate and save report
    report = evaluator.generate_report(results)
    report_path = Path(args.report)
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TEACHER MODEL EVALUATION COMPLETE")
    print("="*60)
    
    compatible_count = sum(1 for r in results.values() 
                          if r.get("success") and r.get("compatible_with_maru"))
    fast_count = sum(1 for r in results.values() 
                    if r.get("success") and r.get("meets_speed_target"))
    
    print(f"Models evaluated: {len(results)}")
    print(f"Compatible with MARU: {compatible_count}")
    print(f"Meet speed target (>500 tok/s): {fast_count}")
    print(f"\nDetailed results: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
