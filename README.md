# Project: Text-to-SQL with Transformers (BART, T5 , GPT-2) on the Spider Dataset

**Status:** Initial Implementation Complete for BART small, Extended to T5 small and GPT-2

## 1. Project Objective

This project implements an end-to-end pipeline for fine-tuning transformer-based models—**BART**, **T5**, and **GPT-2**—for the **Text-to-SQL** task using the **Spider dataset**. Given a natural language question and database schema, the system generates an executable SQL query.

We moved from a notebook-based exploration to a fully modularized and maintainable codebase that cleanly separates concerns for data loading, model training, inference, and evaluation.

## 2. Project File Structure

```
SEQ2SQL/
│
├── .github/
│   └── workflows/        # Placeholder for CI/CD
│
├── data/
│   ├── train_spider.json   # Raw training samples
│   └── tables.json         # Schema definitions
│
├── notebooks/
│   └── 1_eda_and_exploration.ipynb  # Initial analysis
│
├── src/
│   ├── __init__.py
│   ├── config.py           # All config paths and model settings
│   ├── data_pipeline.py    # SpiderDataset class and data loader
│   ├── model_pipeline.py   # Model selection and wrapping
│   └── evaluate.py         # Metric functions
│
├── train.py                # Train any model with config
├── predict.py              # Inference script
├── requirements.txt
└── README.md
```

## 3. Setup and Usage Instructions

### 3.1. Environment Setup

```bash
git clone https://github.com/your-org/seq2sql.git
cd seq2sql

python3 -m venv venv
source venv/bin/activate           # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3.2. Training a Model

```bash
python train.py --model_type bart   # options: bart, t5, gpt2
```

You can adjust the number of training samples inside `train.py` for quick testing vs full training.

### 3.3. Inference with a Trained Model

```bash
python predict.py --model_type t5
```

The model will load from the appropriate folder (e.g., `t5_spider_model/`) and output SQL predictions for sample questions.

## 4. Supported Models & Notes

| Model Type | Pretrained Model         | Architecture Type      | Comments |
|------------|--------------------------|-------------------------|----------|
| `bart`     | `bart-small`             | Encoder-Decoder         | Best for sequence-to-sequence |
| `t5`       | `t5-small`               | Text-to-Text Encoder-Decoder | Schema and question are formatted as a string |
| `gpt2`     | `gpt2`                   | Decoder-Only            | Requires careful input formatting and causal masking |

Specify the model using `--model_type` flag in both training and inference.

## 5. Methodology

### 5.1. Input Format

All models use the same general input format with slight variations depending on architecture:

```
Question: <NL_QUESTION> Schema: <TABLE_NAME: COL1 | COL2 ...>
```

### 5.2. Model-Specific Details

- **BART & T5**: Use Hugging Face's sequence-to-sequence architecture and are compatible with the `Trainer` API.
- **GPT-2**:
  - Handled as an autoregressive decoder-only model.
  - Padding and label shifting are managed carefully for causal language modeling.

## 6. Evaluation Metrics

We compute the following metrics per epoch during training:

- **BLEU Score**: Measures n-gram overlap.
- **Logical Form Accuracy**: Checks exact match with ground-truth SQL (ignores case and whitespace).
- **Exact Match Score**: Requires complete, character-for-character correspondence between compared elements with zero tolerance for any variations or deviations.

Example Output:
```
--- Final Evaluation on 5000 Predictions ---
logical_form_accuracy: 0.2880
bleu: 0.5583
exact_match: 0.1960
```

*(Shown for BART – others will differ.)*

## 7. Automation Readiness (CI/CD)

The `.github/workflows` folder is scaffolded for future CI/CD support. In the future, this could support:

- **Linting & Testing**
- **Model Versioning**
- **Automatic Deployment to Hugging Face Hub**

## 8. Next Steps

- ✅ Add support for T5 and GPT-2 ✅
- 🧪 Implement **execution accuracy** by validating queries on actual DBs.
- 📈 Add experiments with **larger models**: `bart-large`, `t5-base`, etc.
- 🔍 Improve schema linking with graph-based or IR methods.
- 🧰 Add CLI flags for dataset size, batch size, learning rate, etc.
- 📦 Save training checkpoints with automatic evaluation.

## 9. License & Citation

This project is research-oriented. Please cite appropriately when using it in academic or production contexts.

**Maintained by:** Chirag and Utkarsh 
For questions or contributions, raise an issue or submit a PR.
