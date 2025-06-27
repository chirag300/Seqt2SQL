# train.py
from sklearn.model_selection import train_test_split
from transformers import Trainer
from src.data_pipeline import SpiderDataset
from src.model_pipeline import Text2SQLModel
from src.evaluate import compute_metrics
import src.config as config
import json

def main():
    """
    The main function to orchestrate the model training and evaluation pipeline.
    """
    # 1. Initialize the model from the base configuration
    print(f"Initializing model from base: {config.BASE_MODEL_NAME}")
    text2sql_model = Text2SQLModel()

    # --- Data Loading and Splitting ---
    # Load the full dataset first to split it
    with open(config.TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Using a subset of 100 samples for a quick demonstration.
    # For a full run, use the entire `full_data` list.
    subset_data = full_data[:100]

    # Split the data into training and validation sets (90% train, 10% validation)
    train_data, eval_data = train_test_split(subset_data, test_size=0.1, random_state=42)
    
    print(f"Data split into {len(train_data)} training samples and {len(eval_data)} validation samples.")

    # 2. Create the Dataset objects for train and eval
    print("Loading and preparing datasets...")
    train_dataset = SpiderDataset(
        tokenizer=text2sql_model.tokenizer,
        data_list=train_data, # Pass the list of data
        tables_data_path=config.TABLES_DATA_PATH,
        max_length=config.TOKENIZER_MAX_LENGTH
    )
    eval_dataset = SpiderDataset(
        tokenizer=text2sql_model.tokenizer,
        data_list=eval_data, # Pass the list of data
        tables_data_path=config.TABLES_DATA_PATH,
        max_length=config.TOKENIZER_MAX_LENGTH
    )
    print("Datasets prepared successfully.")

    # 3. Update Training Arguments to include evaluation
    config.TRAINING_ARGS.evaluation_strategy = "epoch"
    
    # 4. Initialize the Trainer with evaluation components
    trainer = Trainer(
        model=text2sql_model.model,
        args=config.TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # We need a lambda function to pass the tokenizer to our metrics function
        compute_metrics=lambda p: compute_metrics(p, text2sql_model.tokenizer),
    )

    # 5. Start the training and evaluation process
    print("--- Starting Model Training and Evaluation ---")
    trainer.train()
    print("--- Training Finished ---")

    # 6. Save the fine-tuned model
    text2sql_model.save()

if __name__ == "__main__":
    main()