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
    Supports zero-shot evaluation via --zero_shot flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['bart', 't5'], default='bart',
                        help="Choose which model to evaluate: bart or t5.")
    parser.add_argument('--zero_shot', action='store_true',
                        help="If set, runs zero-shot evaluation with HuggingFace pre-trained model (no fine-tuning).")
    args = parser.parse_args()
    MODEL_TYPE = args.model_type

    # Model selection: fine-tuned vs zero-shot
    if args.zero_shot:
        if MODEL_TYPE == "bart":
            model_dir = "facebook/bart-base"   # You can change to "facebook/bart-small" if available
        else:
            model_dir = "t5-small"
        print(f"Running zero-shot evaluation with model: {model_dir}")
    else:
        if MODEL_TYPE == "bart":
            model_dir = config.MODEL_OUTPUT_DIR
        else:
            model_dir = config.T5_MODEL_OUTPUT_DIR
        print(f"Loading fine-tuned model from: {model_dir}")

    # Load model
    try:
        trained_model = get_model(model_type=MODEL_TYPE, model_name_or_path=model_dir)
        print("Model loaded successfully.")
    except OSError:
        print(f"Error: Model not found at {model_dir}.")
        if not args.zero_shot:
            print("Please run the training script first using: python train.py")
        return

    # Load Spider training data
    with open(config.TRAIN_DATA_PATH, 'r') as f:
        full_data = json.load(f)
    eval_data = full_data[:50]  # Evaluate on first 500 samples

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
