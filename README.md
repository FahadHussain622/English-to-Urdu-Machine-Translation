# English-to-Urdu Machine Translation

## Table of Contents
- Overview
- Dataset
- Preprocessing
- Models
- Evaluation
- Usage
- License
- References

---

## Overview

English-to-Urdu Machine Translation is a deep learning–based project designed to translate English text into Urdu. The project implements multiple sequence-to-sequence architectures, including RNNs, bidirectional RNNs, LSTMs, GRUs, and Transformer-based models. In addition, a pretrained multilingual transformer (mBART-50) is fine-tuned on a dedicated English–Urdu parallel corpus to achieve high-quality translations.

The system provides a complete pipeline covering data preprocessing, model training, evaluation, and deployment. A production-ready REST API enables real-time translation, while all experiments and model performance metrics are tracked using Weights & Biases (W&B).

**Key Highlights**
- Models: RNN, Bi-RNN, LSTM, GRU, Transformer with attention, and fine-tuned mBART-50
- Deployment: Flask-based REST API with Docker support
- Monitoring: Experiment tracking using Weights & Biases

---

## Dataset

The dataset consists of 54,689 parallel English–Urdu sentence pairs. The raw corpus was cleaned and normalized to remove noise and ensure consistency across both languages. Following preprocessing, the dataset was divided into training and testing subsets.

- Size: 54,689 aligned English–Urdu sentence pairs
- Training set: 80% (43,751 pairs)
- Test set: 20% (10,938 pairs)
- Cleaning: Removal of noisy or misaligned sentences and language-specific normalization

---

## Preprocessing

Each sentence pair undergoes multiple preprocessing steps to ensure suitability for training:

- Normalization: Lowercasing and removal of unwanted characters (digits, HTML tags, special symbols)
- Tokenization: Subword tokenization using Byte-Pair Encoding (BPE) or SentencePiece
- Vocabulary Construction: Creation of consistent source (English) and target (Urdu) vocabularies
- Formatting: Conversion to numerical IDs with padding or truncation for batch training

These steps ensure standardized and high-quality inputs for all models.

---

## Models

The following neural machine translation models are implemented and evaluated:

- RNN (Seq2Seq): Encoder–decoder architecture with a unidirectional RNN and attention
- Bi-RNN: Bidirectional RNN encoder with RNN decoder and attention
- LSTM: Encoder–decoder architecture using LSTM units for long-range dependency modeling
- GRU: Encoder–decoder architecture using Gated Recurrent Units
- mBART-50 (Transformer): A pretrained multilingual encoder–decoder transformer fine-tuned on the English–Urdu dataset

All models are implemented in PyTorch. The mBART-50 model is implemented using Hugging Face Transformers. RNN-based models employ additive attention mechanisms.

---

## Evaluation

Translation quality is evaluated using standard automated metrics:

- BLEU: Measures n-gram precision
- ROUGE-L: Measures longest common subsequence recall
- METEOR: Measures unigram precision and recall with synonym matching
- TER: Measures the number of edits required to match reference translations

The fine-tuned mBART-50 model achieves the best performance with:
- BLEU score: 0.45
- ROUGE-L score: 0.69

These results demonstrate the effectiveness of multilingual pretraining for English–Urdu translation.

---

## Usage

### Clone the Repository
```bash
git clone https://github.com/yourusername/english-urdu-translation.git
cd english-urdu-translation
Install Dependencies

(Requires Python 3.8 or higher)

pip install -r requirements.txt

Run the Translation API
python app.py


The API will be available at:

http://localhost:5000


Send a POST request to /translate with the following JSON body:

{"text": "Your English sentence"}


All experiments and training runs are logged using Weights & Biases. To enable tracking, set the WANDB_API_KEY environment variable.

License

This project is licensed under the MIT License. See the LICENSE file for details.

References

Papineni et al. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. ACL.
Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries. ACL.
Banerjee & Lavie (2005). METEOR: An Automatic Metric for MT Evaluation. ACL Workshop.
Snover et al. (2006). Translation Edit Rate with Targeted Human Annotation. AMTA.
Tang et al. (2020). Multilingual Translation with Extensible Multilingual Pretraining and Finetuning. arXiv:2008.00401.
Weights & Biases (W&B). https://wandb.ai




