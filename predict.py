# predict.py
import json
from src.model_pipeline import Text2SQLModel
from src.data_pipeline import get_table_schema
from src.evaluate import compute_metrics
import src.config as config

def main():
    """
    Loads the trained model and runs evaluation over 1000 examples.
    """
    try:
        print(f"Loading fine-tuned model from: {config.MODEL_OUTPUT_DIR}")
        trained_model = Text2SQLModel(model_name_or_path=config.MODEL_OUTPUT_DIR)
        print("Model loaded successfully.")
    except OSError:
        print(f"Error: Model not found at {config.MODEL_OUTPUT_DIR}.")
        print("Please run the training script first using: python train.py")
        return

    # Load Spider training data
    with open(config.TRAIN_DATA_PATH, 'r') as f:
        full_data = json.load(f)
    
    eval_data = full_data[:500]  # evaluate on first 1000 samples

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
