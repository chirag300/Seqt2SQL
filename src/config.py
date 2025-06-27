# src/config.py
import os
from transformers import TrainingArguments

# --- Core Paths ---
# This computes the project root directory dynamically.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "bart_spider_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Data Files ---
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_spider.json")
TABLES_DATA_PATH = os.path.join(DATA_DIR, "tables.json")

# --- Model & Tokenizer ---
BASE_MODEL_NAME = "facebook/bart-base"
TOKENIZER_MAX_LENGTH = 128

# --- Training Configuration ---
# You can customize TrainingArguments here for fine-tuning.
TRAINING_ARGS = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir=f'{RESULTS_DIR}/logs',
    report_to="none",  # Disables wandb integration
)

# --- Prediction/Generation Configuration ---
GENERATION_ARGS = {
    "max_length": 128,
    "num_beams": 4,
    "early_stopping": True
}