import sys,os
import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer

from transformers import AutoTokenizer

NUM_PROCS = os.cpu_count() 
print("NUM_PROCS = " ,NUM_PROCS)

# Load dataset (e.g., CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize the input and target texts
def tokenize_function(examples):
    model_inputs = tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(text_target=examples['highlights'], truncation=True, padding='max_length', max_length=128)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize and prepare the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True,num_proc=NUM_PROCS)

