# Jigsaw Toxic Comment Classification (Multilabel NLP)

This repository implements a multilabel text classification model for predicting the probability of six different types of comment toxicity:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The model is based on **RoBERTa** and built using the **Hugging Face Transformers** library.  
It was evaluated on the Kaggle **Jigsaw Toxic Comment Classification** benchmark.

---

## Approach

### Model Backbone
- Pretrained **RoBERTa (`roberta-base`)**
- Multilabel classification setup
- **Sigmoid activation** for each label
- **Binary cross-entropy loss**

---

### Classification Head Experiments
Two classification head designs were explored:

- **Standard RoBERTa head**
  - Uses the `<s>` (start token) representation for classification

- **Custom pooling head (selected)**
  - Mean pooling over all token embeddings in each sequence
  - Architecture:
    ```
    Dropout → Linear → ReLU → Dropout → Linear
    ```
  - This approach achieved better validation performance and was selected for further experiments

---

## Hyperparameter Tuning

After selecting the custom classification head, hyperparameter tuning was performed under a fixed compute budget:

- **Learning rate tuning**
  - Compared three different learning rates
  - Best learning rate selected based on validation **macro ROC-AUC**

- **Sequence length comparison**
  - Compared maximum sequence lengths of **128 vs 256**
  - Final model uses **128 tokens** for improved computational efficiency with comparable performance

All tuning experiments were conducted with controlled settings to ensure fair comparisons.

---

## Final Training

- Best-performing configuration retrained on the **full training dataset**
- Training performed for **3 epochs**
- Evaluation metric:
  - **Macro ROC-AUC** across all six labels

---

## Results

- **Kaggle Private Leaderboard ROC-AUC:** **0.981**
- **Kaggle Public Leaderboard ROC-AUC:** **0.980**


---

## Repository Contents

- **`RoBERTa_jigsaw_toxic_comment_classification.ipynb`**
  - Complete pipeline including:
    - Data preprocessing
    - Model definition
    - Hyperparameter tuning
    - Final training
    - Error analysis
    - Kaggle submission generation

---

## Tools & Libraries

- Python
- Hugging Face Transformers
- PyTorch
- scikit-learn
- NumPy
- Pandas


