### MODULES ###

import sys,os
import tqdm
import csv
import numpy as np
import pandas as pd
import json
import torch

from datasets import load_dataset
from datasets import load_from_disk

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Load the ROUGE metric
import evaluate

from transformers import AutoTokenizer, BartForConditionalGeneration


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

job_nb = sys.argv[1]
path_to_load =  sys.argv[2] #'/LAB-DATA/GLiCID/users/ibotca@univ-angers.fr/datasets/cnn_dailymail_train'

# Create a directory for storing results
results_dir = f"./results_{job_nb}"  # Ensure correct naming
os.makedirs(results_dir, exist_ok=True)

# Save configuration to the correct directory
config_path = os.path.join(results_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_path}")

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True



NUM_PROCS = os.cpu_count() 
print("NUM_PROCS = " ,NUM_PROCS)

MODEL_HUB = "facebook/bart-large-cnn"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = " , device)


### Load model ###

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_HUB)

print("max_position_embeddings = " , model.config.max_position_embeddings) 

# Load dataset (e.g., CNN/DailyMail)
dataset = load_from_disk(path_to_load)
#dataset = dataset.select(list(range(100))) #selecte a sample



# Define the custom collate function
def collate_fn(batch):
    """
    Custom collate function that add padding for each batch.
    """

    id = [item['id'] for item in batch]

    # Pad the tokenized content
    padded_text_ids = pad_sequence(
        [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id)
    
    padded_text_mask = pad_sequence(
        [torch.tensor(item['input_mask'], dtype=torch.long) for item in batch], 
        batch_first=True, 
        padding_value=0)

    decoder_input_ids = pad_sequence(
        [torch.tensor(item['target_ids'], dtype=torch.long) for item in batch], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id)     
    
    decoder_attention_mask = pad_sequence(
        [torch.tensor(item['target_mask'], 
                      dtype=torch.long) for item in batch], 
                      batch_first=True, 
                      padding_value=0)
    
    input_len = [item['input_len'] for item in batch]

    target_len = [item['target_len'] for item in batch]

    highlights = [item['highlights'] for item in batch]

    
    

    return {
        'id':id,
        'input_ids':padded_text_ids,
        'attention_mask':padded_text_mask,
        'decoder_input_ids':decoder_input_ids,
        'target_mask':decoder_attention_mask,
        'input_len': input_len ,
        'target_len': target_len,
        'highlights': highlights
    }


params = {
    'batch_size': BATCH_SIZE,
    'shuffle': False,
    'collate_fn':collate_fn,
    'num_workers': NUM_LOADER,
    'pin_memory': True  #  Enables faster GPU transfers
    }

# Return batch of data
loader = DataLoader(dataset, **params)

# Return acurate score
rouge = evaluate.load('rouge')


with open(f'./results_{job_nb}/rouge.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["rouge1", "rouge2", "rougeL"]
    writer.writerow(field)

with open(f'./results_{job_nb}/len.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["id", "input_len", "target_len", "generate_len"]
    writer.writerow(field)

model.eval()
model.to(device)

rouge1_score, rouge2_score , rougeL_score = 0, 0, 0
nb_sample = 0

exclude_ids = torch.tensor([0, 1, 2, 3, 50264]).to(device) #spécial token to skip

with torch.no_grad():
    
    for _, batch in tqdm.tqdm(enumerate(loader, 0),desc=f'total iter: {len(loader)}', unit=" iter"):
        

        generated_ids = model.generate(
              input_ids = batch["input_ids"].to(device),
              attention_mask = batch["attention_mask"].to(device), 
              max_length=max_len_resume, 
              num_beams=NUM_BEAM,
              repetition_penalty=repetition_penalty, 
              length_penalty=length_penalty, 
              early_stopping=early_stopping
              )   
        #print(generated_ids)

        generated_txt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        #print(generated_txt)
        #print(type(generated_txt))

        mask = ~torch.isin(generated_ids, exclude_ids) #mask to skip the special tokens 
        generate_len = mask.sum(dim=1)  

        with open(f'./results_{job_nb}/len.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[batch["id"][i], batch["input_len"][i], batch["target_len"][i], generate_len[i].item()] for i in range(len(batch))])       

        # Compute ROUGE scores here
        rouge_results = rouge.compute(predictions=generated_txt, references=batch["highlights"])
        
        
        with open(f'./results_{job_nb}/rouge.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL']])

        rouge1_score += rouge_results['rouge1'].item()
        rouge2_score += rouge_results['rouge2'].item()
        rougeL_score += rouge_results['rougeL'].item()

        nb_sample+=1

with open(f'./results_{job_nb}/rouge_total.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Total_rouge1", "Total_rouge2", "Total_rougeL"]
    writer.writerow(field)
    writer.writerow([rouge1_score/nb_sample*100, rouge2_score/nb_sample*100, rougeL_score/nb_sample*100])
