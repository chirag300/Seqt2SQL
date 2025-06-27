# Project: Text-to-SQL with BART on the Spider Dataset

**Status:** Initial Implementation Complete

## 1. Project Objective

This project implements an end-to-end pipeline for fine-tuning a **BART (Bidirectional and Auto-Regressive Transformer)** model to perform Text-to-SQL translation. Using the **Spider dataset**, our system takes a natural language question as input and generates the corresponding executable SQL query based on the relevant database schema.

The primary focus of this implementation was to establish a modular and maintainable codebase that separates data processing, model training, and evaluation, moving away from a single-notebook approach.

## 2. Project File Structure

The codebase is organized to ensure a clear separation of concerns, making it easier for team members to navigate and contribute.

```
SEQ2SQL/
│
├── .github/
│   └── workflows/        # Placeholder for future CI/CD automation
│
├── data/
│   ├── train_spider.json   # The raw training data used
│   └── tables.json         # Database schema definitions
│
├── notebooks/
│   └── 1_eda_and_exploration.ipynb # EDA notebook for data analysis
│
├── src/
│   ├── __init__.py         # Defines the 'src' directory as a Python package
│   ├── config.py           # Centralized configuration for all paths and parameters
│   ├── data_pipeline.py    # Contains the SpiderDataset class for data handling
│   ├── model_pipeline.py   # An OOP wrapper for our BART model
│   └── evaluate.py         # Functions for calculating model performance metrics
│
├── .gitignore              # Standard git ignore file
├── README.md               # Project documentation (this file)
├── requirements.txt        # Required Python libraries for the project
├── train.py                # Main script to execute the model training pipeline
└── predict.py              # Script for running inference with a trained model
```

## 3. Setup and Usage Instructions

Follow these steps to set up the local environment and run the project pipelines.

### 3.1. Environment Setup

1.  **Clone the repository** to your local machine.
2.  **Navigate to the project's root directory**.
3.  **Create and activate a Python virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

4.  **Install all required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3.2. Running the Pipelines

#### To Train the Model:
The `train.py` script handles the entire fine-tuning process. It will load the data, process it, train the model, and save the final artifacts to the `bart_spider_model/` directory.

```bash
python train.py
```
**Note:** The script is currently configured to run with a small subset of the data (100 samples) for quick testing. This can be adjusted in `train.py`.

#### To Make a Prediction:
After a model has been trained and saved, use the `predict.py` script to test its inference capability.

```bash
python predict.py
```
This script loads the fine-tuned model from `bart_spider_model/` and generates an SQL query for a sample question defined within the script.

## 4. Methodology

### 4.1. Model Architecture
We selected the **BART** architecture for this task, specifically starting from the `facebook/bart-base` pre-trained checkpoint. BART's denoising autoencoder pre-training objective makes it highly effective for sequence-to-sequence tasks, which aligns well with the Text-to-SQL translation problem.

### 4.2. Training Process
The model was fine-tuned on a custom-formatted input that provides the necessary context for query generation.

-   **Input Format:** The model receives a concatenated string containing both the user question and the simplified database schema:
    `Question: <NL_QUESTION> Schema: <TABLE_NAME: COL1 | COL2 ...>`
-   **Training API:** The Hugging Face `Trainer` API was used to manage the fine-tuning loop, which simplifies the process and includes integrated logging and evaluation hooks.

## 5. Performance Evaluation

To assess the model's performance during training, we implemented two key metrics that are computed at the end of each epoch:

1.  **BLEU Score**: This metric evaluates the quality of the generated text by comparing the n-gram overlap between the predicted SQL query and the ground-truth query. It provides a measure of textual similarity.
2.  **Logical Form Accuracy**: This is a strict accuracy metric. It performs a case-insensitive, whitespace-normalized string comparison to check for an exact match between the predicted and ground-truth SQL queries.

The results of these evaluations are printed to the console during the execution of the `train.py` script.

## 6. Automation Readiness (CI/CD)

The project structure includes a `.github/workflows` directory. This is a placeholder to facilitate the future addition of GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD). Potential workflows include:
-   **CI**: Automatically running code linters (e.g., `ruff`) and tests on every push or pull request.
-   **CD**: Automating the model training and deployment process to a model registry like Hugging Face Hub.

These workflows have **not** been implemented yet but the project is structured to support them easily.

## 7. Next Steps

-   **Full Dataset Training**: Scale up the training pipeline to use the complete Spider dataset.
-   **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other parameters to optimize model performance.
-   **Execution Accuracy Metric**: Implement a more robust evaluation metric that involves executing the generated SQL against a database to verify its correctness.
-   **Advanced Models**: Experiment with larger base models (e.g., `bart-large`) or different architectures (e.g., T5, CodeLlama) to compare performance.
