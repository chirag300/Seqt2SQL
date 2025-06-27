# src/model_pipeline.py
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer
from . import config

class Text2SQLModel:
    """
    A pipeline class for initializing, training, and running the Text-to-SQL model.
    """
    def __init__(self, model_name_or_path=config.BASE_MODEL_NAME):
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)

    def train(self, train_dataset):
        """Trains the model on the provided dataset."""
        trainer = Trainer(
            model=self.model,
            args=config.TRAINING_ARGS,
            train_dataset=train_dataset,
        )
        print("--- Starting Model Training ---")
        trainer.train()
        print("--- Training Finished ---")

    def predict(self, question, schema):
        """Generates SQL from a single question and schema string."""
        input_text = f"Question: {question} Schema: {schema}"
        inputs = self.tokenizer(
            [input_text], 
            return_tensors="pt", 
            max_length=config.TOKENIZER_MAX_LENGTH, 
            truncation=True
        )
        
        output_ids = self.model.generate(inputs["input_ids"], **config.GENERATION_ARGS)
        
        sql_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return sql_query

    def save(self, output_dir=config.MODEL_OUTPUT_DIR):
        """Saves the fine-tuned model and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model successfully saved to {output_dir}")