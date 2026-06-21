# Sovereign AI Provenance Auditor

Training-free AI model provenance audit tool for Google Colab T4.

## Overview

This project implements a Colab-compatible audit pipeline for open-weight language models. It uses gradient-based fingerprinting to compare structural similarity between models, tokenizer fertility analysis to measure multilingual token tax, and a governance scorecard to summarize transparency and provenance risk.

The main paper-accurate experiment loads and fingerprints four models live:

- `meta-llama/Llama-3.2-1B`
- `mistralai/Ministral-8B-Instruct-2410`
- `Qwen/Qwen2.5-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

## Files

- `TensorGuard_Lite_Colab.ipynb` - Google Colab notebook with UI.
- `tensorguard_lite_colab.py` - Python source used by the notebook.

## Main Features

- Hugging Face model download and cache support.
- Live four-model paper experiment without stored reference fingerprints.
- Gradient fingerprint extraction from attention and MLP layers.
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
4. Use the `Live Paper Experiment` tab for the full four-model experiment.
5. Paste a Hugging Face token with accepted access for gated models such as Llama.

## Notes

The live paper experiment is slower than the exploratory single-model audit because it downloads and fingerprints all four models sequentially. For a quick app check, use the single-model audit or reduce perturbation runs, sequence length, and sampled gradient entries.

The newer `mistralai/Ministral-3-3B-Instruct-2512` checkpoint uses a multimodal `Mistral3Config`, which is not compatible with the text-only `AutoModelForCausalLM` gradient pipeline. This implementation uses the text-only Ministral family checkpoint listed above for live text fingerprinting.

Technical fingerprints are audit evidence, not standalone legal proof of model origin. Models without white-box weight access remain technically unverifiable under this gradient-based method.
