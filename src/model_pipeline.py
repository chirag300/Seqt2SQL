from transformers import BartTokenizerFast, BartForConditionalGeneration, Trainer
from . import config
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from transformers import T5Tokenizer, T5ForConditionalGeneration

class Text2SQLT5Model:
    """
    A pipeline class for initializing, training, and running the Text-to-SQL model using T5.
    """
    def __init__(self, model_name_or_path="t5-small"):
        print(f"Initializing T5 model from base: {model_name_or_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

    def train(self, train_dataset):
        trainer = Trainer(
            model=self.model,
            args=config.TRAINING_ARGS,
            train_dataset=train_dataset,
        )
        print("--- Starting Model Training (T5) ---")
        trainer.train()
        print("--- Training Finished (T5) ---")

    def predict(self, question, schema):
        # T5 expects task prefix, e.g., "translate SQL: ..."
        input_text = f"translate SQL: {question} Schema: {schema}"
        inputs = self.tokenizer(
            [input_text],
            return_tensors="pt",
            max_length=config.TOKENIZER_MAX_LENGTH,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = self.model.generate(inputs["input_ids"], **config.GENERATION_ARGS)
        sql_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return sql_query

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = config.T5_MODEL_OUTPUT_DIR   # THIS IS THE FIX!
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"T5 Model successfully saved to {output_dir}")

class Text2SQLModel:
    """
    A pipeline class for initializing, training, and running the Text-to-SQL model.
    """
    def __init__(self, model_name_or_path=config.BASE_MODEL_NAME):
        print(f"Initializing model from base: {model_name_or_path}")
        self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

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
        inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to MPS or CUDA/CPU

        output_ids = self.model.generate(inputs["input_ids"], **config.GENERATION_ARGS)
        sql_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return sql_query

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = config.MODEL_OUTPUT_DIR
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model successfully saved to {output_dir}")

def get_model(model_type="bart", model_name_or_path=None):
    """
    Utility to select which model class to instantiate.
    """
    if model_type == "bart":
        return Text2SQLModel(model_name_or_path or config.BASE_MODEL_NAME)
    elif model_type == "t5":
        return Text2SQLT5Model(model_name_or_path or "t5-small")
    else:
        raise ValueError("model_type must be 'bart' or 't5'")
