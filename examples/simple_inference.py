#!/usr/bin/env python3
"""
Simple MARU Inference Example

This script demonstrates how to use a trained MARU model for text generation.
Perfect for getting started with MARU!
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maru import MARU, MARUConfig
from enhanced_mom_gru_config import get_baseline_config
from tokenizer import CharacterTokenizer

class SimpleMARUInference:
    """Simple interface for MARU text generation."""
    
    def __init__(self, checkpoint_path: str = "../checkpoints/narrativeqa_epoch_1_final_converted.pt"):
        """Initialize the inference engine."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 MARU Simple Inference")
        print(f"📱 Device: {self.device}")
        
        # Load model
        self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path: str):
        """Load the MARU model from checkpoint."""
        print(f"📂 Loading model from {checkpoint_path}...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract configuration
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                
                # Create MARU config
                use_enhanced = config_dict.get('use_enhanced_mom_gru', True)
                if 'converted' in checkpoint_path:
                    use_enhanced = False  # Use original MoM-GRU for converted checkpoints

                config = MARUConfig(
                    vocab_size=config_dict.get('vocab_size', 256),
                    d_model=config_dict.get('d_model', 640),
                    hidden_size=config_dict.get('hidden_size', 640),
                    output_dim=config_dict.get('output_dim', 256),
                    memory_size=config_dict.get('memory_size', 256),
                    memory_dim=config_dict.get('memory_dim', 64),
                    num_memories=config_dict.get('num_memories', 6),
                    use_enhanced_mom_gru=use_enhanced,
                    enhanced_mom_gru_config=get_baseline_config() if use_enhanced else None
                )
                
                print(f"📊 Model: {config.vocab_size} vocab, {config.d_model} d_model, {config.hidden_size} hidden")
                
            else:
                print("⚠️ No config found, using defaults")
                config = MARUConfig(vocab_size=256)
            
            # Create and load model
            self.model = MARU(config).to(self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model loaded successfully!")
            else:
                raise ValueError("No model_state_dict found in checkpoint")
            
            self.model.eval()
            
            # Create tokenizer
            self.tokenizer = CharacterTokenizer()
            print(f"📝 Tokenizer ready with {self.tokenizer.vocab_size} tokens")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"🧠 Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text to start generation
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded!")
        
        print(f"🎯 Generating from: '{prompt}'")
        print(f"⚙️ Settings: max_length={max_length}, temperature={temperature}")
        
        # Encode prompt
        input_tensor = self.tokenizer.encode(prompt, max_length=None, padding=False)
        input_ids = input_tensor.tolist()
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            # Get initial hidden state from the prompt
            if input_tensor.shape[1] > 1:
                output, (hidden, memory_state) = self.model(input_tensor, return_hidden=True)
                current_hidden = hidden
                current_memory = memory_state
            else:
                current_hidden = None
                current_memory = None
            
            # Generate tokens one by one
            for step in range(max_length):
                # Get last token
                if len(generated_ids) > 0:
                    last_token = torch.tensor([[generated_ids[-1]]], device=self.device)
                else:
                    last_token = torch.tensor([[32]], device=self.device)  # Space character

                # Generate next token
                if current_hidden is not None and hasattr(self.model, 'generate_step'):
                    output, new_hidden, new_memory = self.model.generate_step(
                        last_token, current_hidden, current_memory
                    )
                    current_hidden = new_hidden
                    current_memory = new_memory
                else:
                    output, (current_hidden, current_memory) = self.model(last_token, return_hidden=True)
                    output = output[:, -1, :]

                # Apply temperature and sample
                logits = output / temperature
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                vocab_size = min(self.tokenizer.vocab_size, logits.shape[-1])
                probs_truncated = probs[:vocab_size]
                probs_truncated = probs_truncated / probs_truncated.sum()
                next_token = torch.multinomial(probs_truncated, 1).item()

                # Check for stopping
                if next_token >= self.tokenizer.vocab_size:
                    break

                generated_ids.append(next_token)
                
                # Show progress every 10 tokens
                if step % 10 == 0 and step > 0:
                    try:
                        current_text = self.tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True)
                        print(f"📝 Step {step}: '{current_text}'")
                    except:
                        pass

        # Decode final result
        try:
            final_tensor = torch.tensor(generated_ids)
            generated_text = self.tokenizer.decode(final_tensor, skip_special_tokens=True)
            print(f"✨ Final result: '{generated_text}'")
            return generated_text
        except Exception as e:
            return f"Generated {len(generated_ids)} tokens but couldn't decode: {e}"

def main():
    """Main function with example usage."""
    print("=" * 60)
    print("🎭 MARU Text Generation Demo")
    print("=" * 60)
    
    # Initialize inference engine
    try:
        maru = SimpleMARUInference()
    except Exception as e:
        print(f"Failed to initialize MARU: {e}")
        return
    
    # Example prompts to try
    example_prompts = [
        "Hello",
        "Once upon a time",
        "The future of AI",
        "In a world where",
        "The scientist discovered"
    ]
    
    print("\n🎯 Running example generations...")
    
    for i, prompt in enumerate(example_prompts, 1):
        print(f"\n--- Example {i} ---")
        try:
            result = maru.generate(prompt, max_length=30, temperature=0.8)
            print(f"✅ Success!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo complete! Try your own prompts:")
    print("   python simple_inference.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
