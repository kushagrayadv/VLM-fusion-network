# From Pixels to Words: Transformer-Based Multimodal Sentiment

This project explores a transformer-based multimodal approach to sentiment analysis by combining visual and textual information from social media content. The model is trained to predict sentiment from image-text pairs, achieving **70.38% accuracy** on the **MVSA-Single** dataset.

## Key Features

- **Multimodal Fusion**: Combines visual features from images and textual features from captions for richer sentiment understanding.
- **Transformer Backbones**:
  - **Text**: [DeBERTa](https://huggingface.co/microsoft/deberta-base) for extracting contextual embeddings from text.
  - **Vision**: [data2vec-vision-base](https://huggingface.co/facebook/data2vec-vision-base) for learning semantic visual features.
- **Cross Attention-based Fusion**: Uses a cross-attention mechanism to integrate visual and textual modalities.
- **Multi-loss Training**: Jointly optimizes sentiment classification loss for image, text and fused modalities to improve generalization.

## Dataset

- **MVSA-Single**: A benchmark dataset containing image-caption pairs labeled with sentiment (`positive`, `negative`, `neutral`).
- Preprocessing includes tokenization (for text) and resizing + normalization (for images).

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- torchvision
- scikit-learn (for evaluation)

## How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt

2. **Train the model**:

   ```bash
   python run.py --epochs=50 --batch_size=16 --num_attention_layers=5

3. **Test the model**:

   ```bash
   python eval.py --batch_size=16