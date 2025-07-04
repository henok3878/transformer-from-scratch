# Transformer From Scratch
> 🚧 **Status:** In developmentment 

A Transformer model implementation from scratch using PyTorch.

## Setup

This project requires conda for managing PyTorch and NumPy dependencies. Other development tools are installed via pip.

### Installation

```bash
git clone https://github.com/henok3878/transformer-from-scratch.git
cd transformer-from-scratch

# CPU version
conda env create -f environment-cpu.yml
conda activate transformer-cpu

# GPU version (with CUDA)
conda env create -f environment-gpu.yml  
conda activate transformer-gpu
```

## Development

### Code Quality Tools

```bash
black src/ tests/     # Format code
isort src/ tests/     # Sort imports  
flake8 src/ tests/    # Lint code
mypy src/            # Type checking
```

### Testing

```bash
pytest                    # Run tests
pytest --cov=transformer # With coverage
```

## Project Structure

```
src/transformer/
├── components/
│   ├── multi_head.py          # Multi-head attention
│   ├── encoder_block.py       # Transformer encoder
│   ├── decoder_block.py       # Transformer decoder
│   ├── input_embedding.py     # Token embeddings
│   ├── positional_encoding.py # Position embeddings
│   ├── feed_forward.py        # Position wise feed forward network
│   └── layer_norm.py          # Layer normalization
├── transformer.py             # Complete model
└── train.py                   # Training script

tests/                         # Unit tests
```

## License

MIT
