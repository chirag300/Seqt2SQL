# predict.py
import argparse
import json
from src.model_pipeline import get_model
from src.data_pipeline import get_table_schema
from src.evaluate import compute_metrics
import src.config as config

def main():
    """
    Loads the trained model (BART or T5) and runs evaluation over 500 examples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['bart', 't5'], default='bart')
    args = parser.parse_args()
    MODEL_TYPE = args.model_type

    # Select correct model output directory based on model type
    if MODEL_TYPE == "bart":
        model_dir = config.MODEL_OUTPUT_DIR
    else:
        model_dir = config.T5_MODEL_OUTPUT_DIR

    print(f"Using model type: {MODEL_TYPE}")
    print(f"Loading fine-tuned model from: {model_dir}")
    try:
        trained_model = get_model(model_type=MODEL_TYPE, model_name_or_path=model_dir)
        print("Model loaded successfully.")
    except OSError:
        print(f"Error: Model not found at {model_dir}.")
        print("Please run the training script first using: python train.py")
        return

    # Load Spider training data
    with open(config.TRAIN_DATA_PATH, 'r') as f:
        full_data = json.load(f)
    
    eval_data = full_data[:100]  # evaluate on first 500 samples

    # Load table schemas
    with open(config.TABLES_DATA_PATH, 'r') as f:
        tables_data = json.load(f)

    # Store predictions and labels
    all_preds = []
    all_labels = []

    print(f"\n--- Running Evaluation on {len(eval_data)} Samples ---")

    for i, example in enumerate(eval_data):
        question = example["question"]
        true_sql = example["query"]
        db_id = example["db_id"]

        try:
            schema = get_table_schema(db_id, tables_data)
            pred_sql = trained_model.predict(question, schema)
        except Exception as e:
            print(f"⚠️ Skipping sample {i} due to error: {e}")
            continue

        all_preds.append(pred_sql)
        all_labels.append(true_sql)

        if i > 0 and i % 100 == 0:
            print(f"Progress: {i} examples processed...")

    # Compute accuracy and BLEU
    metrics = compute_metrics((all_preds, all_labels), trained_model.tokenizer)
    
    print(f"\n--- Final Evaluation on {len(all_preds)} Predictions ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
