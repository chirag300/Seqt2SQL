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
    Computes BLEU and Logical Form Accuracy.
    Handles both token IDs and pre-decoded predictions.
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions safely
    if isinstance(preds[0], (list, np.ndarray)):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    elif isinstance(preds[0], str):
        decoded_preds = preds
    else:
        raise TypeError(f"Unsupported prediction format: {type(preds[0])}")

    # Decode labels safely
    if isinstance(labels[0], (list, np.ndarray)):
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    elif isinstance(labels[0], str):
        decoded_labels = labels
    else:
        raise TypeError(f"Unsupported label format: {type(labels[0])}")

    # # BLEU score
    # smoothie = SmoothingFunction().method4
    # bleu_scores = []
    # for pred, label in zip(decoded_preds, decoded_labels):
    #     pred_tokens = tokenize_sql(pred)
    #     label_tokens = [tokenize_sql(label)]
    #     if pred_tokens:
    #         bleu_scores.append(sentence_bleu(label_tokens, pred_tokens, smoothing_function=smoothie))

    # Logical form accuracy
    correct = sum(
        normalize_sql(p) == normalize_sql(l)
        for p, l in zip(decoded_preds, decoded_labels)
    )

    return {
        'logical_form_accuracy': correct / len(decoded_preds)
    }
