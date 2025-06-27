# predict.py
import json
from src.model_pipeline import Text2SQLModel
from src.data_pipeline import get_table_schema
import src.config as config

def main():
    """
    Loads the trained model and runs a prediction on an example question.
    """
    try:
        # 1. Load the fine-tuned model from the output directory
        print(f"Loading fine-tuned model from: {config.MODEL_OUTPUT_DIR}")
        trained_model = Text2SQLModel(model_name_or_path=config.MODEL_OUTPUT_DIR)
        print("Model loaded successfully.")
    except OSError:
        print(f"Error: Model not found at {config.MODEL_OUTPUT_DIR}.")
        print("Please run the training script first using: python train.py")
        return

    # 2. Define an example question and database
    question = "Find the number of heads of the departments."
    db_id = "department_management"  # An example database from the Spider dataset

    # 3. Load tables data to find the schema
    with open(config.TABLES_DATA_PATH, 'r') as f:
        tables_data = json.load(f)
    schema = get_table_schema(db_id, tables_data)
    
    print(f"\n--- Making a Prediction ---")
    print(f"Database ID: {db_id}")
    print(f"Question: {question}")
    print(f"Schema Used: {schema}")

    # 4. Generate the SQL query
    generated_sql = trained_model.predict(question, schema)
    
    print(f"\nPredicted SQL: {generated_sql}")

if __name__ == "__main__":
    main()