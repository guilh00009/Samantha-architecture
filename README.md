# GPT3

Welcome to the GPT3 repository! This project is an attempt to recreate the architecture and approach from the original OpenAI GPT-3 paper. The repository includes scripts for training, fine-tuning, and inference of a GPT-3-like model using PyTorch and the Hugging Face Transformers library.

### Note: I'm currently working on training these models, now 17M in on it's way. When finished, all weights will be published on huggingface

## Repository Structure

```
gpt3_stable_1m.py




gpt3_stable_with_cross_val.py




gpt3-inference_v2.py




gpt3-test7-SFT.py


```

### Files

- **[gpt3_stable_17m.py](gpt3_stable_17m.py)**: Script for training the GPT-3 model which has approximateley 17,867,008 parameters.
- **[gpt3.py](gpt3.py)**: Script for training the GPT-3 model with cross-validation.
- **[inference.py](inference.py)**: Script for running inference with the trained GPT-3 model.
- **[gpt3-SFT.py](gpt3-SFT.py)**: Script for testing and fine-tuning the GPT-3 model with Supervised Fine-Tuning (SFT).

## Key Features

- **Custom Model Architecture**: Implements custom GPT-3 model components such as [`CustomGPT2Attention`](gpt3-17m.py#L136), [`CustomGPT2MLP`](gpt3-17m.py#L143), [`CustomGPT2Block`](gpt3-17m.py#L150), and [`CustomGPT2LMHeadModel`](gpt3-17m.py#L235).
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
    pip install -U transformers datasetes evaluate torch wandb
    ```

### Training

To train the model, run the following command:

```sh
python gpt3-17m.py


# on MacOS or Linux it's 
python3 gpt3-17m.py

```

### Inference

To generate text using the trained model, run:

```sh
python inference.py


# on MacOS or Linux
python3 gpt3-inference_v2.py
```

### Fine-Tuning

If you have trained a foundation model 
```sh
python gpt3-SFT.py

# on MacOS or Linux
python3 gpt3-SFT.py
```

## Usage

### Training Script

The training script initializes the model, optimizer, and learning rate scheduler. It then enters a training loop where it performs forward and backward passes, applies gradient clipping, and updates the model parameters. The aim of the script is to train a foundation model which can then be fine-tuned for chat / question answering / etc.

### Inference Script

The inference script loads a pre-trained model and tokenizer, moves the model to the appropriate device, and generates text based on user input using the [`generate_text_stream`](inference.py#L246) function.

## Custom Components

Everything was taken from official GPT-2 implementation 

### CustomGPT2Attention

A custom implementation of the GPT-3 attention mechanism with biases included.

### CustomGPT2MLP

A custom implementation of the GPT-3 MLP with biases and standard GeLU activation.

### CustomGPT2Block

A custom implementation of the GPT-3 block with optional pre-layer normalization.

### CustomGPT2LMHeadModel

A custom implementation of the GPT-3 language model head with additional keyword arguments support.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details. Everyone can use and modify this code at their discretion.

## Acknowledgements

Thanks OpenAI, HuggingFace and Pytorch for making this project possible!

- [OpenAI GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)