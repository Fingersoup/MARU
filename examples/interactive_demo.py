#!/usr/bin/env python3
"""
Interactive MARU Demo

This script provides an interactive command-line interface for experimenting
with MARU text generation. Perfect for exploring the model's capabilities!
"""

import sys
import os
import torch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simple_inference import SimpleMARUInference

class InteractiveMARUDemo:
    """Interactive demo for MARU text generation."""
    
    def __init__(self):
        """Initialize the interactive demo."""
        self.maru = None
        self.settings = {
            'max_length': 50,
            'temperature': 0.8
        }
        
    def initialize(self):
        """Initialize the MARU model."""
        print("🚀 Initializing MARU...")
        try:
            self.maru = SimpleMARUInference()
            print("✅ MARU ready for text generation!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize MARU: {e}")
            return False
    
    def show_help(self):
        """Show available commands."""
        print("\n📚 Available Commands:")
        print("  generate <prompt>     - Generate text from prompt")
        print("  set max_length <n>    - Set maximum generation length")
        print("  set temperature <f>   - Set sampling temperature (0.1-2.0)")
        print("  settings              - Show current settings")
        print("  examples              - Show example prompts")
        print("  help                  - Show this help")
        print("  quit                  - Exit the demo")
        print()
    
    def show_settings(self):
        """Show current generation settings."""
        print(f"\n⚙️ Current Settings:")
        print(f"  Max Length: {self.settings['max_length']}")
        print(f"  Temperature: {self.settings['temperature']}")
        print()
    
    def show_examples(self):
        """Show example prompts."""
        examples = [
            "Hello world",
            "Once upon a time",
            "The future of AI",
            "In a world where magic exists",
            "The scientist discovered",
            "Machine learning is",
            "Python programming",
            "Neural networks can",
            "Data science helps",
            "The dragon flew"
        ]
        
        print("\n💡 Example Prompts to Try:")
        for i, example in enumerate(examples, 1):
            print(f"  {i:2d}. {example}")
        print("\nJust type: generate <prompt>")
        print()
    
    def set_setting(self, setting: str, value: str):
        """Set a generation setting."""
        try:
            if setting == 'max_length':
                val = int(value)
                if 1 <= val <= 200:
                    self.settings['max_length'] = val
                    print(f"✅ Max length set to {val}")
                else:
                    print("❌ Max length must be between 1 and 200")
            
            elif setting == 'temperature':
                val = float(value)
                if 0.1 <= val <= 2.0:
                    self.settings['temperature'] = val
                    print(f"✅ Temperature set to {val}")
                else:
                    print("❌ Temperature must be between 0.1 and 2.0")
            
            else:
                print(f"❌ Unknown setting: {setting}")
                print("Available settings: max_length, temperature")
        
        except ValueError:
            print(f"❌ Invalid value for {setting}: {value}")
    
    def generate_text(self, prompt: str):
        """Generate text from a prompt."""
        if not self.maru:
            print("❌ MARU not initialized!")
            return
        
        if not prompt.strip():
            print("❌ Please provide a prompt!")
            return
        
        print(f"\n🎯 Generating from: '{prompt}'")
        print(f"⚙️ Using: max_length={self.settings['max_length']}, temperature={self.settings['temperature']}")
        print("⏳ Generating...")
        
        try:
            result = self.maru.generate(
                prompt, 
                max_length=self.settings['max_length'],
                temperature=self.settings['temperature']
            )
            print(f"\n✨ Generated Text:")
            print(f"'{result}'")
            print()
        
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    def run(self):
        """Run the interactive demo."""
        print("=" * 60)
        print("🎭 MARU Interactive Text Generation Demo")
        print("=" * 60)
        
        # Initialize MARU
        if not self.initialize():
            return
        
        # Show initial help
        self.show_help()
        self.show_settings()
        
        print("💬 Ready for commands! Type 'help' for assistance.")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\n🎮 MARU> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                
                if command == 'quit' or command == 'exit':
                    print("👋 Thanks for using MARU! Goodbye!")
                    break
                
                elif command == 'help':
                    self.show_help()
                
                elif command == 'settings':
                    self.show_settings()
                
                elif command == 'examples':
                    self.show_examples()
                
                elif command == 'generate':
                    if len(parts) > 1:
                        self.generate_text(parts[1])
                    else:
                        print("❌ Usage: generate <prompt>")
                
                elif command == 'set':
                    if len(parts) > 1:
                        set_parts = parts[1].split(None, 1)
                        if len(set_parts) == 2:
                            self.set_setting(set_parts[0], set_parts[1])
                        else:
                            print("❌ Usage: set <setting> <value>")
                    else:
                        print("❌ Usage: set <setting> <value>")
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Goodbye!")
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function."""
    demo = InteractiveMARUDemo()
    demo.run()

if __name__ == "__main__":
    main()
