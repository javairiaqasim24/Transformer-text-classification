# IMDb Sentiment Classification: TF-IDF to Transformers

<p align="center">
  <strong>Comparative NLP study of traditional machine learning and pretrained Transformer models for binary sentiment classification.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface">
  <img src="https://img.shields.io/badge/Task-Sentiment%20Classification-purple">
  <img src="https://img.shields.io/badge/Best%20F1-RoBERTa%2093.30%25-success">
</p>

---

## Overview

This project investigates how **traditional text-classification methods compare with pretrained Transformer models** on the IMDb movie-review sentiment classification task.

The experiment follows a controlled progression:

**TF-IDF → Linear Models → BERT → RoBERTa**

Two classical baselines were trained using TF-IDF features:

- Logistic Regression
- Linear SVM

These were compared with two pretrained Transformer encoders:

- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)

The final held-out test results show that **RoBERTa achieved the strongest overall performance**, reaching **93.23% accuracy and 93.30% F1-score**.

> **Key finding:** RoBERTa improved F1-score by approximately **3.70 percentage points** over the Linear SVM baseline and **1.47 percentage points** over BERT.

---

## Research Question

> **How much performance improvement can pretrained Transformer representations provide over strong TF-IDF-based linear baselines for binary sentiment classification?**

---

## Objectives

- Build a reliable sentiment-classification pipeline.
- Perform systematic dataset quality checks and exploratory analysis.
- Establish strong traditional ML baselines.
- Fine-tune pretrained Transformer models.
- Evaluate all approaches on the same held-out test set.
- Compare Accuracy, Precision, Recall, and F1-score.
- Quantify the improvement obtained by Transformer-based contextual representations.
- Organize the experiment into a reproducible, portfolio-ready project.

---

## Dataset

The project uses the **IMDb Large Movie Review Dataset**, a widely used benchmark for binary sentiment classification.

| Label | Sentiment |
|---:|---|
| `0` | Negative |
| `1` | Positive |

### Experimental split

| Split | Samples |
|---|---:|
| Training | 19,824 |
| Validation | 4,957 |
| Test | 25,000 |
| **Total** | **49,781** |

The test set remained isolated from model development and was used for final evaluation.

---

## Data Quality & EDA

The project included:

- Missing-value analysis
- Duplicate-review analysis
- Class-distribution analysis
- Train/validation/test overlap checks
- Cross-split leakage checks
- Review-length analysis
- Sentiment distribution analysis

Final overlap checks:

```text
Train-Test overlap:        0
Validation-Test overlap:   0
Train-Validation overlap:  0
```

For the analyzed training/validation data:

```text
Mean review length:     ~234 words
Median review length:   ~174 words
Maximum review length:  ~2,470 words
```

The modeling data was approximately balanced between the two sentiment classes.

---

# Methodology

## 1. Traditional NLP Baselines

The classical pipeline was:

```text
Raw Review
    ↓
Text preprocessing
    ↓
TF-IDF Vectorization
    ↓
50,000-feature sparse matrix
    ↓
Linear classifier
    ↓
Sentiment prediction
```

### Logistic Regression

A lightweight linear baseline trained on TF-IDF features.

### Linear SVM

A Linear Support Vector Machine trained using the same TF-IDF representation, providing a controlled comparison with Logistic Regression.

---

# 2. Transformer Models

## BERT

Model:

```text
bert-base-uncased
```

BERT was fine-tuned for binary sequence classification.

## RoBERTa

Model:

```text
roberta-base
```

RoBERTa was fine-tuned for the same downstream sentiment-classification objective.

---

# Experimental Pipeline

```text
                    IMDb Reviews
                         │
                         ▼
              Data Quality Analysis
                         │
                         ▼
             Cleaning & Dataset Split
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          Traditional NLP       Transformer NLP
              │                     │
              ▼                     ▼
             TF-IDF            Tokenization
              │                     │
        ┌─────┴─────┐         ┌─────┴─────┐
        ▼           ▼         ▼           ▼
    Logistic      Linear    BERT       RoBERTa
   Regression      SVM
        │           │         │           │
        └───────────┴─────────┴───────────┘
                         │
                         ▼
                 Held-out Test Set
                         │
                         ▼
          Accuracy / Precision / Recall / F1
                         │
                         ▼
                Model Comparison
```

---

# Results

## Final Test-Set Performance

| Model | Accuracy | Precision | Recall | **F1-score** |
|---|---:|---:|---:|---:|
| Logistic Regression | 89.64% | 89.12% | 90.30% | 89.71% |
| Linear SVM | 89.62% | 89.74% | 89.47% | 89.60% |
| BERT | 91.77% | 91.16% | 92.52% | 91.83% |
| **RoBERTa** | **93.23%** | **92.38%** | **94.24%** | **93.30%** |

### 🏆 Best Model: RoBERTa

- **Accuracy:** 93.23%
- **Precision:** 92.38%
- **Recall:** 94.24%
- **F1-score:** 93.30%

---

## Performance Improvement

Using Linear SVM as the primary traditional-ML reference:

| Comparison | F1 Improvement |
|---|---:|
| BERT vs Linear SVM | **+2.23 pp** |
| RoBERTa vs Linear SVM | **+3.70 pp** |
| RoBERTa vs BERT | **+1.47 pp** |

`pp` = percentage points.

---

## Validation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Linear SVM | 90.64% | 90.23% | 91.24% | 90.73% |
| BERT | 92.19% | 91.91% | 92.61% | 92.26% |
| RoBERTa | 93.14% | 92.45% | 94.01% | 93.23% |

The held-out test results were used as the primary basis for final comparison.

---

# Interpretation

Traditional TF-IDF models achieved approximately **89.6–89.7% F1**, demonstrating that lightweight linear models remain strong baselines.

BERT increased test F1 to **91.83%**, improving over Linear SVM by approximately **2.23 percentage points**.

RoBERTa achieved **93.30% F1**, outperforming Linear SVM by **3.70 percentage points** and BERT by **1.47 percentage points**.

Overall, the experiment supports the conclusion that contextual representations learned through Transformer pretraining provide a measurable advantage for this sentiment-classification task.

---

# Model Trade-offs

| Dimension | TF-IDF + Linear Models | Transformers |
|---|---|---|
| Representation | Sparse lexical features | Contextual representations |
| Training cost | Low | High |
| CPU suitability | Excellent | Limited |
| GPU requirement | Usually unnecessary | Recommended |
| Context understanding | Limited | Strong |
| Engineering complexity | Low | Higher |
| F1 in this experiment | ~89.6–89.7% | ~91.8–93.3% |

The highest-performing model is therefore not necessarily the cheapest model to train or deploy.

---

# Evaluation Metrics

### Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1-score was an important comparison metric because it balances precision and recall.

---

# Project Structure

```text
transformer-text-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_traditional_ml.ipynb
│   ├── 03_transformers_training.ipynb
│   └── 04_model_comparison.ipynb
│
├── results/
│   ├── final_metrics.csv
│   └── figures/
│       ├── model_performance_comparison.png
│       └── f1_comparison.png
│
├── src/
│   ├── preprocessing.py
│   ├── traditional_models.py
│   ├── train_transformer.py
│   └── evaluate.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Notebook Guide

### `01_eda.ipynb`
Dataset loading, data-quality checks, duplicate/leakage analysis, class distribution, review-length analysis, and visualizations.

### `02_traditional_ml.ipynb`
TF-IDF, Logistic Regression, Linear SVM, validation/test evaluation, and classical-model visualizations.

### `03_transformers.ipynb`
Transformer preprocessing, tokenization, BERT fine-tuning, RoBERTa fine-tuning, and recorded validation/test results. This notebook was executed in GPU-enabled Google Colab.

### `04_model_comparison.ipynb`
Consolidated final metrics, model ranking, performance plots, baseline improvements, and final conclusions.

---

# Computational Environment

Traditional ML experiments were developed in a local Python/VS Code environment.

BERT and RoBERTa fine-tuning was performed in **GPU-enabled Google Colab** because Transformer training requires substantially more computational resources.

The Transformer notebook is retained as the record of that experimental workflow.

---

# Reproducibility

The repository contains the data-preparation workflow, EDA, traditional ML experiments, Transformer training workflow, comparison notebook, final metrics, and visualization workflow.

Because hosted Colab GPU sessions are temporary and GPU resources were limited, trained Transformer checkpoints are **not included** in the repository.

This avoids committing large model files while keeping the experimental code and recorded results available.

For a fresh reproduction, `03_transformers.ipynb` can be executed in a GPU-enabled environment.

---

# Limitations

- The dataset is limited to movie reviews, so generalization to other domains is not guaranteed.
- The task contains only positive and negative sentiment classes.
- Extensive hyperparameter optimization was outside the scope of this project.
- Transformer training was constrained by available GPU resources.
- Final Transformer prediction arrays were not retained after the temporary Colab runtime ended, so prediction-level confusion matrices are not reported.
- No synthetic confusion matrices are used; all reported metrics come from completed experimental runs.

---

# Future Work

Possible extensions include:

- DeBERTa, DistilBERT, ELECTRA, or other encoder architectures
- Cross-domain sentiment evaluation
- Robustness testing against spelling errors, slang, negation, and noisy text
- Explainability using SHAP, LIME, or Integrated Gradients
- Inference latency and memory benchmarking
- Accuracy/F1 versus computational-cost analysis
- Deeper qualitative error analysis

---

# Technologies

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Classical ML | Scikit-learn |
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Dataset Handling | Hugging Face Datasets |
| Visualization | Matplotlib, Seaborn |
| Development | VS Code, Jupyter |
| GPU Training | Google Colab |

---

# Key Takeaways

1. **TF-IDF remains a strong and efficient baseline** for binary sentiment classification.
2. Logistic Regression and Linear SVM achieved approximately **89.6–89.7% F1**.
3. **BERT achieved 91.83% F1**.
4. **RoBERTa achieved 93.30% F1**, the best result in the experiment.
5. RoBERTa improved over Linear SVM by approximately **3.70 percentage points** in F1.
6. Transformer models delivered higher predictive performance at a substantially greater computational cost.
7. The experiment demonstrates why modern NLP systems should be evaluated against strong classical baselines rather than in isolation.

---

# Conclusion

This project presents a complete comparative NLP workflow for IMDb sentiment classification, progressing from classical sparse-feature methods to pretrained contextual language models.

The final ranking by test-set F1-score was:

```text
1. RoBERTa              93.30%
2. BERT                 91.83%
3. Logistic Regression  89.71%
4. Linear SVM           89.60%
```

The central finding is that **pretrained Transformer representations provided a meaningful performance advantage over TF-IDF-based linear models, with RoBERTa achieving the strongest overall performance**.

At the same time, the classical baselines remained competitive and required substantially less computational effort, highlighting the practical trade-off between model performance and computational cost.

---

## Author

**Javairia Qasim**

NLP / Machine Learning project focused on comparative evaluation of classical text classification and pretrained Transformer architectures.
