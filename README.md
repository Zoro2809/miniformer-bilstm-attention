# Mini-Former: BiLSTM Encoder with Luong Attention Decoder
## Encoder-Decoder Models With and Without Attention — Comparative Study

---

## Assignment Details
- **Assignment**: Assignment 6 - Encoder-Decoder Models with and without Attention
- **Task**: Review, Implementation, and Comparative Analysis of Encoder-Decoder Models with and without Attention Mechanism
- **Paper**: Efficient Machine Translation with a BiLSTM-Attention Approach (arXiv:2410.22335)
- **Name**: Harsh Kumar
- **PRN**: 202301100054
- **Batch**: DL-4

---

## Live Notebook
- **Google Colab**: Run MiniFormer_Complete.ipynb on Google Colab with T4 GPU
- **Original Paper**: https://arxiv.org/abs/2410.22335
- **Original Code**: https://github.com/mindspore-lab/models/tree/master/research/arxiv_papers/miniformer

---

## Paper Reference
- **Title**: Efficient Machine Translation with a BiLSTM-Attention Approach
- **Authors**: Yuxu Wu, Yiren Xing
- **Published**: arXiv:2410.22335 | October 29, 2024
- **Domain**: Natural Language Processing — Machine Translation
- **Original Framework**: MindSpore 2.2.14

---

## Overview

This repository contains the complete TensorFlow 2.19.0 implementation of the Mini-Former architecture proposed in the research paper "Efficient Machine Translation with a BiLSTM-Attention Approach" by Yuxu Wu and Yiren Xing (arXiv:2410.22335, October 2024). The original paper implements Mini-Former using the MindSpore framework developed by Huawei. Since MindSpore is not compatible with Google Colab T4 GPU, this repository re-implements the exact same architecture in TensorFlow 2.19.0 for full reproducibility.

The assignment covers a complete pipeline from paper review to code implementation, training, evaluation, and comparative analysis of encoder-decoder models with and without the attention mechanism on an English-German translation task.

---

## Paper Summary

The Mini-Former model proposes a novel Seq2Seq architecture that:
- Uses a Bidirectional LSTM (BiLSTM) as the encoder to capture both forward and backward context
- Uses an LSTM decoder with Luong-style scaled dot-product attention to dynamically focus on relevant source positions
- Introduces learnable initial hidden and cell states for the encoder initialized from U(-0.01, 0.01)
- Achieves better BLEU scores than the full Transformer on WMT14 English-German benchmark
- Uses only 40 percent of the Transformer model size making it suitable for resource-constrained deployment

Original paper results on WMT14 English-German:

Metric     | Transformer | Mini-Former
-----------|-------------|------------
BLEU-1     | 0.39        | 0.42
BLEU-2     | 0.20        | 0.22
BLEU-3     | 0.12        | 0.14
BLEU-4     | 0.07        | 0.09
ROUGE-1 F1 | 0.47        | 0.49
ROUGE-2 F1 | 0.23        | 0.25
ROUGE-L F1 | 0.45        | 0.46
Model Size | 100%        | 40%

---



## Model Architecture

### With Attention Model — Mini-Former
Source Text (English)
|
[Embedding Layer]  (vocab_size x 128)
|
[BiLSTM Encoder]  (256 units bidirectional = 512 output dim)
|
[Linear Projection]  (512 -> 256 for decoder initial states)
|
[LSTM Decoder]  <----  [Luong Attention over all encoder states]
|
[Dense Output Layer]  (256 -> target vocab size)
|
[Softmax]
|
Target Text (German)

### Without Attention Model — Baseline
Source Text (English)
|
[Embedding Layer]
|
[LSTM Encoder]  (unidirectional, final hidden state only)
|
[Fixed Context Vector]  (256 dim, entire source compressed here)
|
[LSTM Decoder]  (no attention, uses fixed context only)
|
[Dense Output Layer]
|
Target Text (German)

### Attention Formula — Luong Scaled Dot Product
AttnScores = softmax( (Dec_out x Enc_out) / sqrt(d) )
Context    = AttnScores x Enc_outputs
Output     = tanh( Linear( concat(Context, Dec_hidden) ) )


---

## Hyperparameters

Parameter           | Value | Source
--------------------|-------|-----------------------------
Embedding dimension | 128   | train_seq2seqsum.py (paper)
Hidden units        | 256   | train_seq2seqsum.py (paper)
Encoder type        | BiLSTM| Seq2Seq.py (paper)
Attention type      | Luong | Seq2Seq.py (paper)
Optimizer           | Adam  | utils.py (paper)
Learning rate       | 0.001 | train_seq2seqsum.py (paper)
Batch size          | 64    | train_seq2seqsum.py (paper)
Gradient clip norm  | 5.0   | utils.py (paper)
Max sequence length | 20    | this implementation
Training epochs     | 15    | this implementation
Beam search size    | 50    | decode.py (paper)

---

## Dataset

### Original Paper Dataset
- Name: WMT14 English-German
- Source: Workshop on Machine Translation 2014
- Size: Millions of bilingual sentence pairs
- Sources: European corpus, UN corpus, News Commentary corpus
- Split: 4:1 train to test ratio
- Download: https://aistudio.baidu.com/datasetdetail/1999

### This Implementation Dataset
- Name: opus_books English-German
- Source: Helsinki-NLP/opus_books via Hugging Face Datasets
- Total pairs available: 51,467
- Total pairs after filtering: 9,930
- Train split: 7,944 sentences (80 percent)
- Test split: 1,986 sentences (20 percent)
- Max sentence length: 20 tokens
- Source vocabulary size: 14,177 words
- Target vocabulary size: 16,462 words

---

## Official GitHub Repository Files Studied

The official MindSpore implementation was studied from:
https://github.com/mindspore-lab/models/tree/master/research/arxiv_papers/miniformer

File                 | Purpose
---------------------|--------------------------------------------------
Seq2Seq.py           | MiniFormer and Decoder class with attention
data.py              | GigaDataset class for WMT14 data loading
train_seq2seqsum.py  | Main training script with Adam optimizer
decode.py            | Beam search inference with beam size 50
eval.py              | BLEU and ROUGE evaluation script
utils.py             | Trainer class with training loop and checkpointing

---

## Results

### Part 2 — With Attention Model on Sample Dataset (30 sentences, 100 epochs)

Metric      | Score
------------|-------
Final Loss  | 0.0091
BLEU-1      | 1.0000
BLEU-2      | 1.0000
BLEU-3      | 0.9100
BLEU-4      | 0.5800
Corpus BLEU | 0.8522
Accuracy    | 100%
GPU         | Tesla T4
Framework   | TensorFlow 2.19.0

### Part 3 — Comparison on Large Dataset (9930 sentences, 15 epochs)

Metric                  | Without Attention | With Attention
------------------------|-------------------|---------------
Final Loss              | 4.2890            | 4.1376
BLEU-1                  | 0.0888            | 0.0911
BLEU-2                  | 0.0137            | 0.0199
BLEU-3                  | 0.0089            | 0.0093
BLEU-4                  | 0.0111            | 0.0124
Training Time (seconds) | 466.0             | 872.7
Model Parameters        | 8,941,006         | 9,925,326
Encoder Type            | Simple LSTM       | BiLSTM
Attention Mechanism     | None              | Luong
Output Quality          | Repetitive        | More aligned

---

## Key Observations

1. The with-attention model achieves lower final loss of 4.1376 compared to 4.2890 for the baseline, confirming that attention improves model convergence.

2. The with-attention model outperforms the baseline across all four BLEU metrics on 1,986 unseen test sentences, validating the paper's core claim about attention improving translation quality.

3. The without-attention model shows repetitive output patterns such as repeating "und john und john" and "nicht zu gehen" across different source sentences. This is a direct consequence of the fixed context vector bottleneck which cannot retain all source information for variable length sequences.

4. The with-attention model requires approximately 87 percent more training time due to attention computation at each decoder step. This trade-off is justified by consistent improvement in translation quality across all metrics.

5. Our BLEU scores are lower than the paper's reported results due to smaller dataset size (7,944 vs millions of sentences), fewer training epochs (15 vs full convergence), word-level tokenization instead of subword tokenization, and greedy decoding instead of beam search with beam size 50.

6. The parameter difference between both models is 984,320 parameters — these are the attention projection layer, concatenation layer, and associated weights. This shows the attention mechanism adds a modest number of parameters but delivers meaningful quality improvements.

---

## References
- Wu, Y. and Xing, Y. (2024). Efficient Machine Translation with a BiLSTM-Attention Approach. arXiv:2410.22335
- Bahdanau, D. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. arXiv:1409.0473
- Luong, M. T. (2015). Effective Approaches to Attention-Based Neural Machine Translation. arXiv:1508.04025
- Vaswani, A. (2017). Attention is All You Need. arXiv:1706.03762
- Sutskever, I., Vinyals, O. and Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. NeurIPS 2014
- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Hugging Face Datasets: https://huggingface.co/datasets/Helsinki-NLP/opus_books
- Original MindSpore Code: https://github.com/mindspore-lab/models/tree/master/research/arxiv_papers/miniformer
