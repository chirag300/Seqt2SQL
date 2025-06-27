# src/data_pipeline.py
import json
import torch
from torch.utils.data import Dataset

def get_table_schema(db_id, tables_data):
    """Extracts a simplified schema string for a given db_id."""
    for table in tables_data:
        if table["db_id"] == db_id:
            # Joining column names for a simplified schema representation
            columns = " | ".join(col[1] for col in table["column_names_original"])
            return f"{table['table_names_original'][0]}: {columns}"
    return "unknown_schema"

class SpiderDataset(Dataset):
    """
    A Dataset class to handle the Spider data preparation for the model.
    It tokenizes the data upon initialization.
    """
    # MODIFICATION: Change the signature to accept a 'data_list'
    def __init__(self, tokenizer, data_list, tables_data_path, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load tables data
        with open(tables_data_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)

        # Prepare inputs and outputs from the provided list
        self.inputs = [
            f"Question: {q['question']} Schema: {get_table_schema(q['db_id'], tables_data)}"
            for q in data_list
        ]
        self.outputs = [q['query'] for q in data_list]
        
        # Tokenize the data
        self._tokenize()

    def _tokenize(self):
        """Tokenizes the inputs and outputs."""
        self.encodings = self.tokenizer(
            self.inputs, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        )
        labels = self.tokenizer(
            self.outputs, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
        ).input_ids
        
        # In training, the model ignores token IDs of -100 in the loss calculation.
        labels[labels == self.tokenizer.pad_token_id] = -100
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item