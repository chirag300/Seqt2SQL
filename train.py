import argparse
from sklearn.model_selection import train_test_split
from transformers import Trainer
from src.data_pipeline import SpiderDataset
from src.model_pipeline import get_model
from src.evaluate import compute_metrics
import src.config as config
import json

def main():
    # ---- Argument parsing ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['bart', 't5', 'gpt2'], default='bart')  # Added GPT-2
    args = parser.parse_args()
    MODEL_TYPE = args.model_type

    # 1. Initialize the model
    print(f"Initializing model type: {MODEL_TYPE}")
    text2sql_model = get_model(model_type=MODEL_TYPE)

    # --- Data Loading and Splitting ---
    with open(config.TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    subset_data = full_data[:2000]
    train_data, eval_data = train_test_split(subset_data, test_size=0.1, random_state=42)
    print(f"Data split into {len(train_data)} training samples and {len(eval_data)} validation samples.")

    # 2. Create the Dataset objects for train and eval
    print("Loading and preparing datasets...")
    train_dataset = SpiderDataset(
        tokenizer=text2sql_model.tokenizer,
        data_list=train_data,
        tables_data_path=config.TABLES_DATA_PATH,
        max_length=config.TOKENIZER_MAX_LENGTH
    )
    eval_dataset = SpiderDataset(
        tokenizer=text2sql_model.tokenizer,
        data_list=eval_data,
        tables_data_path=config.TABLES_DATA_PATH,
        max_length=config.TOKENIZER_MAX_LENGTH
    )
    print("Datasets prepared successfully.")

    config.TRAINING_ARGS.evaluation_strategy = "epoch"

    # 3. Initialize the Trainer with evaluation components
    trainer = Trainer(
        model=text2sql_model.model,
        args=config.TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, text2sql_model.tokenizer)
    )

    # 4. Start training and evaluation
    print("--- Starting Model Training and Evaluation ---")
    trainer.train()
    print("--- Training Finished ---")

    # 5. Save the fine-tuned model
    text2sql_model.save()

if __name__ == "__main__":
    main()
