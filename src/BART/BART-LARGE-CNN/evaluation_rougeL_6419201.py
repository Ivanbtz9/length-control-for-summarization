### MODULES ###

import sys,os
import tqdm
import csv
import numpy as np
import pandas as pd
import json
import torch

from datasets import load_dataset

# Load the ROUGE metric
import evaluate
from rouge_score import rouge_scorer

### CONFIG ###

# Load configuration from JSON
with open('./config.json', 'r') as f:
    config = json.load(f)

# Assign values from config (with defaults to avoid crashes if a key is missing)
SEED = config.get("SEED", 0)  # Random seed
NUM_LOADER = config.get("NUM_LOADER", 50)  # Number of threads
max_len = config.get("max_len", 1024)  # Max input length for encoder
BATCH_SIZE = config.get("BATCH_SIZE", 64)
NUM_BEAM = config.get("NUM_BEAM", 5)
max_len_resume = config.get("max_len_resume", 200)  # Max summary length
repetition_penalty = config.get("repetition_penalty", 2.0)  # Penalty for repetitive text
length_penalty = config.get("length_penalty", 1.0)  # Length penalty in beam search
early_stopping = config.get("early_stopping", True)  # Stop decoding early


NUM_PROCS = os.cpu_count() 
print("NUM_PROCS = " ,NUM_PROCS)


# Return acurate score
rouge = evaluate.load('rouge')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


dataset = load_dataset("csv", data_files="./results_6419201_len_and_txt/len_and_txt.csv")["train"]

# Define the evaluation function
def evaluation(examples):
    """
    Compute ROUGE scores for a batch of generated texts with the rouge object from evaluate
    """
    generated_txt = examples["generated_txt"]  # List of generated summaries
    references = examples["references_txt"]  # List of reference summaries

    # Ensure lists have the same length
    if len(generated_txt) != len(references):
        raise ValueError(f"Mismatch in number of predictions ({len(generated_txt)}) and references ({len(references)})")

    # Compute ROUGE scores
    rouge_results = rouge.compute(
        predictions=generated_txt,
        references=references,
        use_stemmer=True
    )

    # Convert scalar values to lists
    return {key: [value] * len(generated_txt) for key, value in rouge_results.items()}

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Define the evaluation function
def evaluation_rouge_score(examples):
    """
    Compute ROUGE scores for a batch of generated texts withe the rouge object from rouge_score
    """
    generated_txt = examples["generated_txt"]  # List of generated summaries
    references = examples["references_txt"]  # List of reference summaries

    scores = scorer.score(generated_txt ,references)

    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure)
    }

# Apply evaluation with batch processing
dataset = dataset.map(evaluation_rouge_score, batched=False, num_proc=NUM_PROCS)

dataset.to_pandas()[['id','rouge1', 'rouge2', 'rougeL']].to_csv("./results_6419201_len_and_txt/rouge_score.csv",index=False)
pd.DataFrame(dataset.to_pandas()[['rouge1', 'rouge2', 'rougeL']].mean(axis=0)).transpose().to_csv("./results_6419201_len_and_txt/total_rouge_score.csv",index=False)