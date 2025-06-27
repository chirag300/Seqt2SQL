# src/evaluate.py
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def tokenize_sql(sql_query):
    """A simple tokenizer for SQL queries."""
    if not isinstance(sql_query, str):
        return []
    # Add spaces around parentheses and operators for better tokenization
    sql_query = sql_query.replace("(", " ( ").replace(")", " ) ")
    return sql_query.split()

def normalize_sql(sql):
    """Normalizes SQL queries for logical form comparison."""
    if not isinstance(sql, str):
        return ""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)  # Replace multiple whitespaces with one
    sql = sql.strip().replace(" ;", ";") # Remove space before semicolon
    return sql

def compute_metrics(eval_preds, tokenizer):
    """
    This function is designed to be used by the Hugging Face Trainer.
    It computes BLEU score and Logical Form Accuracy.
    """
    preds, labels = eval_preds
    
    # preds are the model's raw output logits, get the most likely token
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode generated tokens into text, skipping special tokens like <pad>
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # --- Metric Calculation ---
    
    # 1. BLEU Score
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = tokenize_sql(pred)
        label_tokens = [tokenize_sql(label)] # BLEU score expects a list of reference translations
        if pred_tokens: # Cannot compute BLEU for empty predictions
            bleu_scores.append(sentence_bleu(label_tokens, pred_tokens, smoothing_function=smoothie))

    # 2. Logical Form Accuracy
    correct_predictions = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        if normalize_sql(pred) == normalize_sql(label):
            correct_predictions += 1

    # Prepare results dictionary
    result = {
        'bleu_score': np.mean(bleu_scores) if bleu_scores else 0.0,
        'logical_form_accuracy': correct_predictions / len(decoded_preds)
    }
    
    return result