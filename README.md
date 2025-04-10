# GPT3

Welcome to the GPT3 repository! This project is an attempt to recreate the architecture and approach from the original OpenAI GPT-3 paper. The repository includes scripts for training, fine-tuning, and inference of a GPT-3-like model using PyTorch and the Hugging Face Transformers library.

## Repository Structure

- **[train-17m.py](train-17m.py)**: Script for training the GPT-3 model which has 17M (17,867,008) parameters.
- **[train_125m.py](train-125m.py)**: Script for training the GPT-3 model with cross-validation.
- **[inference.py](inference.py)**: Script for running inference with the trained GPT-3 model.
- **[fine-tune-SFT.py](fine-tune-SFT.py)**: Script for testing and fine-tuning the GPT-3 model with Supervised Fine-Tuning (SFT).

## Key Features

- **Custom Model Architecture**: Implements custom GPT-3 model components such as [`CustomGPT2Attention`](train-17m.py#L136), [`CustomGPT2MLP`](train-17m.py#L143), [`CustomGPT2Block`](train-17m.py#L150), and [`CustomGPT2LMHeadModel`](train-17m.py#L235).
- **Training Loop**: Includes gradient accumulation, gradient clipping, and perplexity computation.
- **Inference**: Supports text generation stream with top-k and top-p filtering.
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
    git clone https://github.com/krll-corp/GPT3.git
    cd GPT3
    ```

2. Install the required packages:
    ```sh
    pip install -U transformers datasets evaluate torch wandb
    ```

### Training

To train the model, run the following command:

```sh
# on Windows use
python train-17m.py

# on MacOS or Linux it's 
python3 train-17m.py
```
#### Note: training is very memory-consuming task, if you don't have that much VRAM, you can "trade" compute to memory by reducing batch size and increasing gradient accumulation steps

### Inference

I've published three checkpoints on Huggingface:
- https://huggingface.co/k050506koch/GPT3-dev - 17m checkpoint.
- https://huggingface.co/k050506koch/GPT3-dev-125m - 125m checkpoint trained on M2 Mac with batch_size=12 and no grad_accum.
- https://huggingface.co/k050506koch/GPT3-dev-125m-0612 - 125m checkpoint trained longer on rtx 4090 with batch_size=12 and grad_accum=4.
- https://huggingface.co/k050506koch/GPT3-dev-125m-1202 - the newest checkpoint (also 125m) which was trained without a cloud (M2 MacBook Air, batch size=12; grad_accum steps were from 4 to 16 (increased to the end when loss started plateuting)). This checkpoint scores 27.31% on MMLU which is a little higher than 

Note for future: currently I’m training another checkpoint with tweaked hyperparameters so it will be even smarter.

To inference one of HuggingFace checkpoints, you can set model_path="k050506koch/GPT3-dev-125m-0612". Don't forget to set trust_remote_code=True because all checkpoints have custom architecture.

I have also published gguf checkpoints in same repositories on HuggingFace, so they can be inferred with LM Studio or llama.cpp:

```sh
lms get k050506koch/GPT3-dev-125m-0612
```

To generate text, run:

```sh
python inference.py

# on MacOS or Linux
python3 inference.py
```

### Fine-Tuning

If you have trained a foundation model, then you may want to Supervised Fine-Tune it for chat or any other purpose.

For this, i use HuggingFace TRL package, you should install it before fine-tuning:

```sh
pip install trl
```

Then, start the training:

```sh
# on Windows
python fine-tune-SFT.py

# on MacOS or Linux
python3 fine-tune-SFT.py
```

## Usage

### Training Script

The training script initializes the model, optimizer, and learning rate scheduler. It then enters a training loop where it performs forward and backward passes, applies gradient clipping, and updates the model parameters. The aim of the script is to train a foundation model which can then be fine-tuned for chat / question answering / etc.

### Inference Script

The inference script loads a pre-trained model and tokenizer, moves the model to the appropriate device, and generates text based on user input using the [`generate_text_stream`](inference.py#L254) function.

You can also download weights from HuggingFace:

17 million parameters: https://huggingface.co/k050506koch/GPT3-dev (architecture demonstrator)

125 million parameters: https://huggingface.co/k050506koch/GPT3-dev-125m-1202 (more like an actual implementation, the most recent model)

To use the model, you have to paste a path to the folder inside [model_path = ""](inference.py#L319). This will work for both HuggingFace-based models of **same architecture** and locally-based ones. To download a remote-hosted model, type "k050506koch/GPT3-dev-125m-1202" into [model_path = ""](inference.py#L319).

#### Note: All these models are highly undertrained due to my computational budget. Some of them underwent 600,000 training steps with a batch size of 12 and a sequence length of 512, totaling approximately 3.7 billion tokens (512 * 12 * 600,000 = 3,686,400,000). (This is a very small amount for the model of this size. For example, OpenAI trained their GPT-3 model series on 300 Bn tokens.)

#### The most recent one underwent 60,000 steps (1/10th of total training) and already shows better performance and coherence than others. Curently I am still training it, i think this will lead to even better results.

## Custom Components

**All custom components are based on the official GPT-2 implementation and have been modified according to the GPT-3 paper.**

- CustomGPT2Attention: GPT-3 attention mechanism with biases.

- CustomGPT2MLP: GPT-3 MLP with biases and standard GeLU.

- CustomGPT2Block: GPT-3 block with pre-layer normalization (can be switched back to GPT-2's post-layer normalization).

- CustomGPT2LMHeadModel: GPT-3 language model head with keyword arguments support.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details. Everyone can use and modify this code at their discretion.

## Acknowledgements

Thanks OpenAI, HuggingFace and Pytorch for making this project possible!

- [OpenAI GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
