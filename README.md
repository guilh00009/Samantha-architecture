# Samantha

Welcome to the Samantha repository! This project implements a custom GPT-3-like architecture called "Samantha" for training large language models. The repository includes scripts for training, fine-tuning, and inference of Samantha models using PyTorch and the Hugging Face Transformers library.

## Repository Structure

- **[train-17m.py](train-17m.py)**: Script for training the small Samantha model with ~17M parameters.
- **[train-125m.py](train-125m.py)**: Script for training the medium Samantha model with ~125M parameters.
- **[train-350m.py](train-350m.py)**: Script for training the large Samantha model with ~350M parameters.
- **[inference.py](inference.py)**: Script for running inference with the trained Samantha models.
- **[fine-tune-SFT.py](fine-tune-SFT.py)**: Script for testing and fine-tuning the Samantha models with Supervised Fine-Tuning (SFT).

## Key Features

- **Samantha Architecture**: Implements custom GPT-3-like model components with pre-layer normalization and biases.
- **Multiple Model Sizes**: Support for training models ranging from 17M to 350M parameters.
- **Efficient Training**: Includes gradient accumulation, gradient clipping, and perplexity computation.
- **Text Generation**: Supports high-quality text generation with top-k and top-p filtering.
- **Logging and Checkpointing**: Uses Weights & Biases for logging and saves model checkpoints periodically.

## Getting Started

### Prerequisites

- Python 3.8+ (I used 3.12)
- PyTorch (Stable or Nightly)
- Transformers (Hugging Face)
- Datasets
- Weights & Biases (wandb)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/guilh00009/Samantha-architecture.git
    cd Samantha-architecture
    ```

2. Install the required packages:
    ```sh
    pip install -U transformers datasets evaluate torch wandb
    ```

### Training

To train different sized Samantha models, run the appropriate training script:

```sh
# Small model (~17M parameters)
python train-17m.py

# Medium model (~125M parameters)
python train-125m.py

# Large model (~350M parameters) - Recommended for best performance
python train-350m.py
```

#### Note: Training is memory-intensive. For larger models, you can trade compute for memory by reducing batch size and increasing gradient accumulation steps. The 350M model requires significant VRAM (at least 16GB recommended).

### Inference

The Samantha models support high-quality text generation. You can use the inference scripts with locally trained models or load them from Hugging Face.

To run inference with a trained Samantha model:

```sh
python inference.py
```

Make sure to update the `model_path` variable in the inference script to point to your trained model directory.

#### Model Specifications:
- **Samantha-17M**: n_embd=256, n_layer=6, n_head=4 (17M parameters)
- **Samantha-125M**: n_embd=768, n_layer=12, n_head=12 (125M parameters)
- **Samantha-350M**: n_embd=1024, n_layer=24, n_head=16 (350M parameters)

All models use pre-layer normalization and biases for improved training stability and performance.

### Fine-Tuning

After training a foundation Samantha model, you can fine-tune it for specific tasks like chat, question answering, or code generation.

#### Fine-tuning Datasets (Combined)

**Custom ShareGPT Dataset**
- **Dataset**: `merged_dataset_project_human_sharegpt (2).jsonl`
- **Format**: Custom JSONL with conversation format
- **Content**: Human-AI conversation pairs
- **Size**: ~30k conversation samples

**No Robots Dataset**
- **Dataset**: `HuggingFaceH4/no_robots`
- **Usage**: Additional Supervised Fine-Tuning data
- **Splits**: `train` and `test`
- **Purpose**: Diverse instruction tuning examples

**Combined Dataset Features:**
- **Total Training Samples**: ~40k+ conversations
- **Diverse Conversation Types**: Both custom and standardized formats
- **Enhanced Model Capabilities**: Better generalization across different conversation styles

#### Multi-GPU Training (H100 Support)

The fine-tuning script is optimized for multi-GPU setups:

**Hardware Configuration:**
- **GPUs**: 2x H100 (40GB VRAM each)
- **Total VRAM**: 80GB
- **Recommended Setup**: DataParallel or DistributedDataParallel

**Memory-Optimized Configuration:**
- **Batch Size**: 4 per GPU (conservative for 40GB VRAM)
- **Gradient Accumulation**: 8 steps (effective batch size: 64)
- **Mixed Precision**: BF16/FP16 support for faster training
- **Memory Efficient**: Streaming datasets to minimize RAM usage
- **Parallel Processing**: Multi-worker data loading
- **Gradient Checkpointing**: Enabled for memory efficiency

First, install the required packages:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install trl transformers datasets wandb accelerate
```

Then, start fine-tuning:

```sh
# Single GPU (basic usage)
python fine-tune-SFT.py

# Multi-GPU with H100 optimization (recommended for 2x H100 40GB setup)
torchrun --nproc_per_node=2 fine-tune-SFT.py \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-4

# Balanced configuration (good performance vs memory)
torchrun --nproc_per_node=2 fine-tune-SFT.py \
    --batch_size=6 \
    --gradient_accumulation_steps=6 \
    --learning_rate=1.5e-4

# Memory-conservative configuration (if running into OOM)
torchrun --nproc_per_node=2 fine-tune-SFT.py \
    --batch_size=2 \
    --gradient_accumulation_steps=16 \
    --learning_rate=8e-5
```

## Usage

### Training Scripts

The training scripts initialize the Samantha model, optimizer, and learning rate scheduler. They then enter a training loop with gradient accumulation, clipping, and logging. The goal is to train foundation models that can be fine-tuned for various downstream tasks.

### Monitoring and Troubleshooting

#### H100 GPU Monitoring
```sh
# Monitor GPU usage
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv -l 1

# Monitor training progress
tail -f wandb/latest-run/logs/*.log
```

#### Performance Tips
- **Memory Issues**: Reduce batch size or increase gradient accumulation steps
- **Slow Training**: Ensure TF32 is enabled and NCCL is properly configured
- **Dataset Issues**: Check that both dataset files are present and accessible
- **GPU Utilization**: Monitor with `nvidia-smi` and adjust batch sizes accordingly

#### Expected Performance (2x H100 40GB)
- **Training Speed**: ~300-600 samples/second (depending on configuration)
- **Memory Usage**: ~30-35GB per GPU (with gradient checkpointing)
- **Effective Batch Size**: 64-128 (across both GPUs)
- **Training Time**: ~6-12 hours for full fine-tuning (3 epochs)
- **Memory Efficiency**: Optimized for 40GB VRAM per GPU

### Inference Script

The inference script loads a trained Samantha model and generates high-quality text. Update the `model_path` variable to point to your trained model directory.

#### Model Performance Notes:
Training Samantha models requires significant computational resources. The models are designed to achieve good performance with proper training data and compute. The 350M parameter model provides the best balance of performance and computational requirements for most applications.

## Samantha Architecture Components

**The Samantha architecture is based on the GPT-3 paper with custom modifications for improved training stability and performance.**
All custom components are based on the official GPT-2 implementation and have been modified according to the GPT-3 paper.

CustomGPT2Attention: GPT-3 attention mechanism with biases.

CustomGPT2MLP: GPT-3 MLP with biases and standard GeLU.

CustomGPT2Block: GPT-3 block with pre-layer normalization (can be switched back to GPT-2's post-layer normalization).

CustomGPT2LMHeadModel: GPT-3 language model head with keyword arguments support.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Thanks to the open-source community for making this project possible!

- [OpenAI GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
