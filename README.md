# Language Constrained Multimodal Hyper Adapter (LCMHA)

This repository contains the official implementation of the paper:

> **Language Constrained Multimodal Hyper Adapter For Many-to-Many Multimodal Summarization**
> *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)*

**LCMHA** improves many-to-many multimodal summarization (M3S) by using a **Language Constrained Hypernetwork** to generate language-specific adapter weights. This allows the model to balance shared multimodal knowledge with language-specific patterns, preventing interference between languages.

## 🌟 Key Features
* **Hypernetwork-based Adapters:** Generates dynamic weights for adapters based on source/target language embeddings.
* **Language Regularization:** A classification loss ensures the hypernetwork learns distinct language representations.
* **Multimodal Fusion:** Integrates visual features (Faster R-CNN) with text using a joint multimodal adapter.


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
```

### 2. Project Structure
```bash
hparams/: Argument definitions (Model, Data, LoRA, HyperAdapter).

transformers/: Customized Transformers library (Required).

train.py: Main training script.

test.py: Inference and evaluation script.

run_train.sh: Example training script.

run_test.sh: Example inference script.

utils.py: Utility functions.
```

📂 Data Preparation
Text Data
Prepare your dataset in a text file. 

🚀 Training
To train the model (e.g., using LLaMA-3-8B as backbone), you can use the provided run_train.sh script.

Important Arguments:
```bash
* --use_hyper true : Activates the Hyper Adapter module.
* --hyper_classification true : Enables the language regularization task
* --hyper_classification_loss_ratio 0.4: The weight ($\alpha$) for the regularization loss.
* --lora_train_hyper true: Trains the hypernetwork alongside LoRA adapters.
```

Bash
```bash
bash run_train.sh
```

Manual Command:
```bash
python train.py \
    --train_dataset ./data/train.txt \
    --model_name_or_path /path/to/llama-3-8b \
    --output_dir ./output_dir \
    --do_train true \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 3.0 \
    --use_hyper true \
    --lora_train_hyper true \
    --use_img true \
    --hyper_classification true \
    --hyper_classification_loss_ratio 0.4
  ``` 
📊 Inference & Evaluation

To generate summaries and calculate ROUGE scores, use run_test.sh.
```bash
bash run_test.sh
```
Manual Command:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name_or_path /path/to/base_model \
    --test_checkpoint ./checkpoint/your_best_checkpoint \
    --test_dataset ./data/test_data/xx-xx.txt \
    --output_prediction_path ./results \
    --use_hyper true \
    --hyper_predict true \
    --use_img true
``` 
The script will automatically compute ROUGE-1, ROUGE-2, and ROUGE-L scores after generation.

📝 Citation
If you use this code or dataset in your work, please cite our ACL 2025 paper:
```
@inproceedings{liu-etal-2025-language,
    title = "Language Constrained Multimodal Hyper Adapter For Many-to-Many Multimodal Summarization",
    author = "Liu, Nayu  and Yao, Fanglong  and Luo, Haoran  and Yang, Yong  and Tang, Chen  and
      Lv, Bo",
      editor = "Che, Wanxiang  andNabende, Joyce  andShutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics      (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2025.acl-long.1229",
    pages = "25285--25298",
    ISBN = "979-8-89176-251-0",
  
}
```
