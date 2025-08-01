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
BASE_MODEL_NAME = "lucadiliello/bart-small"  # Default model, change as needed
TOKENIZER_MAX_LENGTH = 128

# --- Training Configuration ---
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
}

# --- Add support for T5 ---
T5_MODEL_NAME = "t5-small"  # You can change to "google/flan-t5-small" etc.
T5_MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "t5_spider_model")

# --- Add support for GPT-2 ---
GPT2_MODEL_NAME = "gpt2"  # You can change to "gpt2-medium", "gpt2-large", etc.
GPT2_MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "gpt2_spider_model")

