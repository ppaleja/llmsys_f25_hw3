# MiniTorch: A Deep Learning Framework with Transformer Support

MiniTorch is a minimal deep learning framework built from scratch for educational purposes. It implements core neural network components including automatic differentiation, tensor operations, CUDA acceleration, and a full decoder-only transformer architecture (GPT-2 style) for natural language processing tasks.

## Key Features

- **Custom Tensor Operations**: CPU and CUDA-accelerated tensor operations including matrix multiplication, element-wise operations, and reductions
- **Automatic Differentiation**: Complete autodiff system for gradient computation
- **Neural Network Modules**: Modular building blocks including Linear layers, Dropout, LayerNorm, Embeddings, and more
- **Transformer Architecture**: Full implementation of decoder-only transformer with multi-head attention, feed-forward networks, and pre-LN architecture
- **NLP Pipelines**: Ready-to-use pipelines for sentiment analysis and machine translation
- **CUDA Support**: GPU acceleration through custom CUDA kernels for improved performance


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA toolkit (optional, for GPU acceleration)
- NVIDIA GPU with compatible drivers (optional)

### Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.extra.txt
   pip install -r requirements.txt
   ```

2. **Install MiniTorch in development mode:**
   ```bash
   pip install -e .
   ```

3. **Compile CUDA kernels (optional, for GPU support):**
   ```bash
   bash compile_cuda.sh
   ```

### CUDA Setup Notes

If using CUDA acceleration, ensure your CUDA toolkit version is compatible with your PyTorch installation. For CUDA 12.1:

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch CUDA support:
```python
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
``` 

## Architecture Overview

### Core Framework Components

MiniTorch provides a complete deep learning framework with the following components:

#### Tensor Operations (`minitorch/tensor_ops.py`, `minitorch/cuda_kernel_ops.py`)
- Custom tensor implementation with automatic differentiation support
- CPU and GPU (CUDA) backend support
- Operations include: matrix multiplication, element-wise operations, reductions, broadcasting
- CUDA kernels for: `mapKernel`, `zipKernel`, `reduceKernel`, `MatrixMultiplyKernel`

#### Neural Network Functions (`minitorch/nn.py`)
- **`logsumexp`**: Numerically stable log-sum-exp computation
- **`softmax_loss`**: Combined softmax and cross-entropy loss for classification tasks
- **`softmax`**: Standard softmax activation
- **`GELU`**: Gaussian Error Linear Unit activation

#### Basic Neural Network Modules (`minitorch/modules_basic.py`)

1. **`Linear`**: Fully connected linear transformation layer
2. **`Dropout`**: Regularization layer that randomly zeros elements during training
3. **`LayerNorm1d`**: Layer normalization for stabilizing training
4. **`Embedding`**: Learned word embeddings for NLP tasks

### Transformer Architecture (`minitorch/modules_transformer.py`)

The framework implements a complete decoder-only transformer based on the GPT-2 architecture:

#### `MultiHeadAttention`
Implements masked multi-head self-attention with:
- Projection layers for queries, keys, and values
- Scaled dot-product attention with causal masking
- Multi-head attention with configurable number of heads
- Output projection layer

The attention mechanism computes:
$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is a causal mask preventing attention to future positions.

#### `FeedForward`
Position-wise feed-forward network with:
- Two linear transformations with GELU activation
- Configurable hidden dimension expansion

#### `TransformerLayer`
Complete transformer block using pre-LN (pre-layer normalization) architecture:
- Multi-head self-attention with residual connection
- Feed-forward network with residual connection
- Layer normalization before each sub-layer

#### `DecoderLM`
Full language model combining:
- Token embeddings
- Positional embeddings
- Multiple transformer layers
- Final layer normalization and output projection

## Usage

### Running Tests

The framework includes comprehensive tests for all components:

```bash
# Test tensor functions
python -m pytest -l -v -k "test_logsumexp_student"
python -m pytest -l -v -k "test_softmax_loss_student"

# Test basic modules
python -m pytest -l -v -k "test_linear_student"
python -m pytest -l -v -k "test_dropout_student"
python -m pytest -l -v -k "test_layernorm_student"
python -m pytest -l -v -k "test_embedding_student"

# Test transformer modules
python -m pytest -l -v -k "test_multihead_attention_student"
python -m pytest -l -v -k "test_transformer_layer_1_student"
python -m pytest -l -v -k "test_transformer_layer_2_student"
python -m pytest -l -v -k "test_decoder_lm_student"

# Run all tests
python -m pytest -l -v
```

### Sentiment Analysis

Run sentiment classification on text data:

```bash
python project/run_sentiment_linear.py
```

### Machine Translation

Train a transformer model for German-to-English translation on the IWSLT14 dataset:

```bash
python project/run_machine_translation.py
```

The machine translation pipeline includes:
- Automatic dataset loading and preprocessing (IWSLT14 De-En)
- Training with configurable hyperparameters
- Validation and testing with BLEU score evaluation
- Generation using argmax decoding

**Expected Performance:**
- BLEU score ~7 after first epoch
- BLEU score ~20 after 10 epochs
- Training time: ~1 hour per epoch on A10G GPU
- ~25 seconds per training step

Results are saved in `./workdir_vocab10000_lr0.02_embd256/`.

## Project Structure

```
minitorch/
├── autodiff.py           # Automatic differentiation
├── tensor.py             # Tensor class and operations
├── tensor_ops.py         # CPU tensor operations
├── cuda_kernel_ops.py    # CUDA kernel operations
├── cuda_ops.py           # CUDA backend support
├── operators.py          # Basic mathematical operators
├── nn.py                 # Neural network functions
├── modules_basic.py      # Basic NN modules
├── modules_transformer.py # Transformer architecture
├── module.py             # Base module class
├── optim.py              # Optimizers
└── datasets.py           # Dataset utilities

project/
├── run_sentiment_linear.py      # Sentiment analysis pipeline
└── run_machine_translation.py   # Machine translation pipeline

src/
└── combine.cu            # CUDA kernel implementations

tests/
├── test_nn_student.py                    # NN function tests
├── test_modules_basic_student.py         # Basic module tests
├── test_modules_transformer_student.py   # Transformer tests
└── data/                                 # Test data
```

## Technical Details

### Data Types
- Tensor storage uses `numpy.float32` for efficiency
- CUDA operations handle memory allocation, deallocation, and host-device transfers

### Backend Support
Most modules accept a `backend` argument to switch between CPU and CUDA implementations:
```python
from minitorch.cuda_kernel_ops import CudaKernelOps

# CPU backend (default)
model = DecoderLM(...)

# CUDA backend
backend = CudaKernelOps()
model = DecoderLM(..., backend=backend)
```

### Dropout Implementation
Dropout uses `np.random.binomial` for mask generation to ensure reproducibility across different random seeds.

## References

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask) - GPT-2 architecture
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745.pdf) - Pre-LN vs Post-LN discussion
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Jurafsky and Martin

## Acknowledgments

This project was developed as part of a Large Language Model Systems course. The framework demonstrates key concepts in deep learning including automatic differentiation, neural network architectures, and GPU acceleration.

