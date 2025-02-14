### MODULES ###

import sys,os
import tqdm
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from datasets import load_from_disk

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Load the ROUGE metric
import evaluate

from transformers import AutoTokenizer, BartForConditionalGeneration


### CONFIG ###

path_to_save = '/LAB-DATA/GLiCID/users/ibotca@univ-angers.fr/datasets/cnn_dailymail'

NUM_PROCS = os.cpu_count() 

print("NUM_PROCS = " ,NUM_PROCS)

MODEL_HUB = "facebook/bart-large-cnn"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = " , device)

max_len = 1024 # max embedding number for the encoder part

BATCH_SIZE =64

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_HUB)

print("max_position_embeddings = " , model.config.max_position_embeddings) 

del model

# Load dataset (e.g., CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split='train')


def len_distrib(batch):

    len_articles = []
    len_highlights = []
    
    # Prefix the "summarize: " instruction to each article (can be adjusted depending on your task)
    batch["article"] = ["summarize: " + article for article in batch["article"]]

    for article, highlight in zip(batch["article"], batch["highlights"]):
        len_articles.append(len(tokenizer(article, truncation=False)["input_ids"]))
        len_highlights.append(len(tokenizer(highlight, truncation=False)["input_ids"]))


    source = tokenizer(batch["article"],truncation=True, max_length=max_len)
    resume = tokenizer(batch["highlights"],truncation=True, max_length=max_len)

    return {
        'input_ids': source['input_ids'], 
        'input_mask': source['attention_mask'],
        'input_len': len_articles,
        'target_ids': resume['input_ids'], 
        'target_mask': resume['attention_mask'],
        'target_len': len_highlights
        }



dataset = dataset.map(len_distrib,num_proc=NUM_PROCS,batched=True,batch_size=64)# Save the Hugging Face dataset
dataset.save_to_disk(path_to_save)
print("Dataset saved successfully.")