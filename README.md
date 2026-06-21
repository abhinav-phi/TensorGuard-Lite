# Sovereign AI Provenance Auditor

Training-free AI model provenance audit tool for Google Colab T4.

## Overview

This project implements a Colab-compatible audit pipeline for open-weight language models. It uses gradient-based fingerprinting to compare structural similarity between models, tokenizer fertility analysis to measure multilingual token tax, and a governance scorecard to summarize transparency and provenance risk.

The main live experiment loads and fingerprints the current open-weight comparison set:

- `meta-llama/Llama-3.2-1B`
- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `Qwen/Qwen2.5-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

## Files

- `TensorGuard_Lite_Colab.ipynb` - Google Colab notebook with UI.
- `tensorguard_lite_colab.py` - Python source used by the notebook.

## Main Features

- Hugging Face model download and cache support.
- Live multi-model experiment without stored reference fingerprints.
- Exact 16-feature gradient fingerprint extraction from global, attention, FFN/MLP, embedding, and structural signals.
- Optional Gaussian perturbation during gradient probing.
- Pairwise cosine similarity and Euclidean distance matrix.
- PCA lineage visualization.
- English, Hindi, and Tamil tokenizer fertility analysis.
- Token tax and approximate attention-cost multiplier.
- Tokenizer vocabulary Jaccard similarity matrix.
- Safetensors and architecture metadata reporting.
- Sovereign transparency and disclosure scorecard.

## Running in Google Colab

1. Open `TensorGuard_Lite_Colab.ipynb` in Google Colab.
2. Set runtime to `T4 GPU`.
3. Run the notebook cell.
4. Use the `Live Paper Experiment` tab for the live multi-model experiment.
5. Paste a Hugging Face token with accepted access for gated models such as Llama.

## Notes

The live paper experiment is slower than the exploratory single-model audit because it downloads and fingerprints multiple models sequentially. For a quick app check, use the single-model audit or reduce perturbation runs, sequence length, and sampled gradient entries.

Mistral/Ministral is temporarily removed from the live experiment. The current 3B Ministral checkpoints use a newer config path that is not compatible with this text-only `AutoModelForCausalLM` gradient pipeline, while the compatible 8B checkpoint is too heavy for reliable Colab T4 gradient runs. `HuggingFaceTB/SmolLM2-1.7B-Instruct` is used as the additional lightweight open-weight comparison model.

Technical fingerprints are audit evidence, not standalone legal proof of model origin. Models without white-box weight access remain technically unverifiable under this gradient-based method.
