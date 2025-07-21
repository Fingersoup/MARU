# MARU Examples

This directory contains example scripts and demos for using the MARU (Memory-Augmented Recurrent Unit) architecture.

## 🚀 Quick Start

### 1. Simple Inference Example

The simplest way to get started with MARU text generation:

```bash
cd examples
python simple_inference.py
```

This script demonstrates:
- Loading a trained MARU model
- Basic text generation
- Multiple example prompts
- Performance monitoring

**Example Output:**
```
🎯 Generating from: 'Hello'
✨ Final result: 'Hello mory bhes at unees scriptiagr'
```

### 2. Interactive Demo

For hands-on experimentation with MARU:

```bash
cd examples
python interactive_demo.py
```

This provides an interactive command-line interface where you can:
- Try different prompts
- Adjust generation settings (temperature, max_length)
- See real-time generation
- Experiment with various parameters

**Example Session:**
```
🎮 MARU> generate Once upon a time
🎯 Generating from: 'Once upon a time'
✨ Generated Text: 'Once upon a time script witiptang use of weript...'

🎮 MARU> set temperature 1.2
✅ Temperature set to 1.2

🎮 MARU> generate The future of AI
🎯 Generating from: 'The future of AI'
✨ Generated Text: 'The future of AI will bring amazing possibilities...'
```

## 📋 Available Commands (Interactive Demo)

| Command | Description | Example |
|---------|-------------|---------|
| `generate <prompt>` | Generate text from prompt | `generate Hello world` |
| `set max_length <n>` | Set max generation length | `set max_length 100` |
| `set temperature <f>` | Set sampling temperature | `set temperature 0.9` |
| `settings` | Show current settings | `settings` |
| `examples` | Show example prompts | `examples` |
| `help` | Show help | `help` |
| `quit` | Exit demo | `quit` |

## ⚙️ Generation Parameters

### Temperature
Controls creativity vs. coherence:
- **0.1-0.5**: More focused, coherent text
- **0.6-0.9**: Balanced creativity and coherence
- **1.0-2.0**: More creative, diverse text

### Max Length
Controls how many tokens to generate:
- **10-30**: Short completions
- **50-100**: Medium paragraphs
- **100+**: Longer text (may become repetitive)

## 💡 Example Prompts to Try

### Creative Writing
- "Once upon a time"
- "In a world where magic exists"
- "The dragon flew over"
- "The mysterious door opened"

### Technical Topics
- "Machine learning is"
- "Neural networks can"
- "The algorithm works by"
- "Data science helps"

### Conversational
- "Hello, how are you"
- "What is the meaning of"
- "Can you explain"
- "The best way to"

### Story Starters
- "The scientist discovered"
- "In the year 2050"
- "The last human on Earth"
- "When the robots awakened"

## 🔧 Customization

### Using Your Own Model

To use a different checkpoint:

```python
from simple_inference import SimpleMARUInference

# Load your custom model
maru = SimpleMARUInference("path/to/your/checkpoint.pt")

# Generate text
result = maru.generate("Your prompt here", max_length=50, temperature=0.8)
print(result)
```

### Batch Generation

For generating multiple texts:

```python
prompts = ["Hello", "The future", "Once upon"]
results = []

for prompt in prompts:
    result = maru.generate(prompt, max_length=30, temperature=0.8)
    results.append(result)
    print(f"'{prompt}' -> '{result}'")
```

### Advanced Settings

```python
# More creative generation
creative_result = maru.generate("Tell me a story", max_length=100, temperature=1.2)

# More focused generation
focused_result = maru.generate("Explain AI", max_length=50, temperature=0.3)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found"**
   - Ensure the checkpoint file exists
   - Check the path is correct
   - Make sure you're in the right directory

2. **"CUDA out of memory"**
   - Reduce max_length
   - Use CPU instead: set `device='cpu'` in the code

3. **"Tokenizer errors"**
   - Ensure you're using the CharacterTokenizer
   - Check that the model was trained with the same tokenizer

4. **"Generation seems repetitive"**
   - Increase temperature (try 0.9-1.2)
   - Reduce max_length
   - Try different prompts

### Performance Tips

- **GPU**: Use CUDA for faster generation
- **Batch Size**: Keep batch size at 1 for inference
- **Memory**: Monitor GPU memory usage
- **Temperature**: Start with 0.8 and adjust based on results

## 📊 Expected Performance

With the provided checkpoint:
- **Speed**: ~40ms per inference on GPU
- **Memory**: ~37MB GPU memory
- **Quality**: Coherent short text, creative patterns
- **Vocabulary**: 256 character-level tokens

## 🎯 Next Steps

1. **Experiment** with different prompts and settings
2. **Train** your own MARU model on custom data
3. **Integrate** MARU into your applications
4. **Contribute** improvements and examples

## 📚 Additional Resources

- [Main README](../README.md) - Project overview
- [Architecture Guide](../docs/ARCHITECTURE.md) - Technical details
- [Training Guide](../train_individual_datasets.py) - How to train models
- [Validation Tests](../validation_tests.py) - Model testing

---

Happy experimenting with MARU! 🎉
