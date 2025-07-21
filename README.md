# MARU: Memory-Augmented Recurrent Unit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MARU** (Memory-Augmented Recurrent Unit) is a novel neural architecture that combines the strengths of state-space models (Mamba) and memory-augmented recurrent networks to achieve efficient long-sequence modeling with **O(1) inference complexity**.

## 🚀 Key Features

- **Novel Architecture**: Combines Mamba blocks with Memory-augmented GRU cells
- **O(1) Inference**: Constant-time autoregressive generation
- **Memory-Augmented**: Multiple persistent memory matrices with selective access
- **Two-Stage Processing**: Parallel Mamba processing followed by sequential memory-augmented recurrence
- **Efficient Training**: Supports long sequences with reasonable memory requirements

## 🏗️ Architecture Overview

MARU processes sequences in two complementary stages:

1. **Stage 1 - Mamba Block**: Processes the entire sequence in parallel using selective state-space modeling to capture long-range dependencies
2. **Stage 2 - MoM-GRU**: Processes Mamba output recurrently with memory-augmented GRU cells that maintain persistent memory across timesteps

This design achieves the best of both worlds: the parallel efficiency of state-space models and the persistent memory capabilities of recurrent networks.

```
Input → Embedding → Mamba Block → MoM-GRU → Output
                      ↓              ↓
                 Long-range      Persistent
                Dependencies     Memory
```

## 📊 Current Results

Our trained MARU model (6.7M parameters) demonstrates:
- **Coherent text generation** with phonetic understanding
- **Grammar awareness** and contextual responses
- **Genre recognition** (e.g., fairy tale patterns from "Once upon")
- **Technical vocabulary** handling

### Example Outputs
```
Input: "Hello" → Output: "Hellong ual thescris eror"
Input: "The" → Output: "These Datatl.ue><mer i"
Input: "Once upon" → Output: "Once upons eBook ily atind, M"
Input: "AI" → Output: "AIcext"><heread tins"
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Quick Install
```bash
git clone https://github.com/yourusername/maru.git
cd maru
pip install -r requirements.txt
```

### Dependencies
```bash
pip install torch>=2.0.0 numpy>=1.21.0 transformers>=4.21.0
```

## 🚀 Quick Start

### Inference
```python
import torch
from src.maru import MARU, MARUConfig
from src.tokenizer import CharacterTokenizer

# Load trained model
model = torch.load('checkpoints/narrativeqa_epoch_1_final_converted.pt')
tokenizer = CharacterTokenizer()

# Generate text
prompt = "Hello"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    output = model.generate(input_ids, max_length=20, temperature=0.8)
    generated_text = tokenizer.decode(output[0].tolist())
    print(f"Generated: {generated_text}")
```

### Training
```python
from train_individual_datasets import main

# Train on NarrativeQA dataset
python train_individual_datasets.py --datasets narrativeqa --max-length 512
```

## 📁 Project Structure

```
maru/
├── src/                          # Core source code
│   ├── maru.py                   # Main MARU architecture
│   ├── mamba_block.py            # Mamba state-space model
│   ├── mom_gru_cell.py           # Memory-augmented GRU cell
│   ├── enhanced_mom_gru_cell.py  # Production MoM-GRU with stabilization
│   └── tokenizer.py              # Character-level tokenizer
├── checkpoints/                  # Trained model checkpoints
├── scripts/                      # Utility scripts
├── train_individual_datasets.py  # Training script
├── quick_inference_test.py       # Inference testing
└── requirements.txt              # Dependencies
```

## 🔬 Technical Details

### Model Configuration
- **Vocabulary Size**: 256 (character-level)
- **Model Dimension**: 640
- **Parameters**: ~6.7M
- **Memory Matrices**: 4 persistent memory banks
- **Memory Size**: 128 slots per bank

### Architecture Components
- **Mamba Block**: Selective state-space model with data-dependent transitions
- **MoM-GRU Cell**: Memory-augmented GRU with multiple memory matrices
- **Router Network**: Selects appropriate memory banks for read/write operations
- **Memory Operations**: Attention-based addressing with read/write heads

## 📚 Documentation

- [Architecture Details](docs/ARCHITECTURE.md) - Deep dive into the technical architecture
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [Training Guide](docs/TRAINING.md) - How to train MARU models
- [API Reference](docs/API.md) - Complete API documentation

## 🧪 Testing

Run the quick inference test:
```bash
python quick_inference_test.py
```

Run comprehensive tests:
```bash
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on Mamba architecture by Gu & Dao (2023)
- Inspired by memory-augmented neural networks
- Built with PyTorch

---

**Note**: This is an experimental architecture. While showing promising results, it's still under active development and research.
