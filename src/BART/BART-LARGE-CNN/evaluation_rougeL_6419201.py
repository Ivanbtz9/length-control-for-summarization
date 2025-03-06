### MODULES ###

import sys,os
import tqdm
import csv
import numpy as np
import pandas as pd
import json
import torch

# Load the ROUGE metric
import evaluate

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


NUM_PROCS = os.cpu_count() 
print("NUM_PROCS = " ,NUM_PROCS)


# Return acurate score
rouge = evaluate.load('rouge')


with open(f'./results_6419201_len_and_txt/rouge.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    writer.writerow(field)


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

        


        #print(generated_txt)
        #print(type(generated_txt))

        mask = ~torch.isin(generated_ids, exclude_ids) #mask to skip the special tokens 
        generate_len = mask.sum(dim=1)  

        with open(f'./results_{job_nb}/len.csv', 'a', newline='') as file:
            writer = csv.writer(file)
    
            min_len = min(len(batch["id"]), len(generate_len))

            for i in range(min_len):
                try:
                    writer.writerow([batch["id"][i], batch["input_len"][i], batch["target_len"][i], generate_len[i].item()])
                except Exception as e:
                    print(f"Error writing row {i}: {e}")
                    
        # Compute ROUGE scores here
        generated_txt = [text.lower().strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        references = [ref.lower().strip() for ref in batch["highlights"]]
        rouge_results = rouge.compute(predictions=generated_txt,
                                      references=references,
                                      use_stemmer=True  # Ensures correct ROUGE comparison
                                      )
        
        
        with open(f'./results_{job_nb}/rouge.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL']])

        rouge1_score += rouge_results['rouge1']* min_len
        rouge2_score += rouge_results['rouge2']* min_len
        rougeL_score += rouge_results['rougeL']* min_len

        nb_sample+=  min_len

with open(f'./results_{job_nb}/rouge_total.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Total_rouge1", "Total_rouge2", "Total_rougeL"]
    writer.writerow(field)
    writer.writerow([rouge1_score/nb_sample, rouge2_score/nb_sample, rougeL_score/nb_sample])
