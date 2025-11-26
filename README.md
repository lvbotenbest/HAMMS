# Language Constrained Multimodal Hyper Adapter (LCMHA)

This repository contains the official implementation of the paper:

> **Language Constrained Multimodal Hyper Adapter For Many-to-Many Multimodal Summarization**
> *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)*

**LCMHA** improves many-to-many multimodal summarization (M3S) by using a **Language Constrained Hypernetwork** to generate language-specific adapter weights. This allows the model to balance shared multimodal knowledge with language-specific patterns, preventing interference between languages.

## 🌟 Key Features
* **Hypernetwork-based Adapters:** Generates dynamic weights for adapters based on source/target language embeddings.
* **Language Regularization:** A classification loss ensures the hypernetwork learns distinct language representations.
* **Multimodal Fusion:** Integrates visual features (Faster R-CNN) with text using a joint multimodal adapter.
* **Flexible Backbones:** Supports LLaMA-3 (via LoRA), mT5, and mBART.

## 🛠️ Installation

### 1. Environment Setup
**Crucial Step:** This project relies on a **customized version of the transformers library** provided in this repository (located in the `./transformers` folder). You **must** install this local version instead of the official PyPI version to enable the hypernetwork features.

```bash
# 1. Create a conda environment
conda create -n lcmha python=3.9
conda activate lcmha

# 2. Install the LOCAL customized transformers library
# (Ensure you are in the project root directory containing the 'transformers' folder)
pip install -e ./transformers

# 3. Install other dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install peft rouge_score nltk h5py datasets tqdm
