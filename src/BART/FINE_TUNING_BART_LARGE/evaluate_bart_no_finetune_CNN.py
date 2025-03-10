

### MODULES ###

import sys,os
import tqdm
import csv
import numpy as np
import pandas as pd
import json
import torch

from datasets import load_dataset

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Load the ROUGE metric
import evaluate

from transformers import AutoTokenizer, BartForConditionalGeneration


### CONFIG ###

# Load configuration from JSON
with open('./config_finetune_bart_large.json', 'r') as f:
    config = json.load(f)


SEED = config['config_machine']["SEED"]
NUM_LOADER = config['config_machine']["NUM_LOADER"] #depends of the number of thread 
BATCH_SIZE = config['config_machine']["BATCH_SIZE"]

NUM_PROCS = os.cpu_count() 
print("NUM_PROCS = " ,NUM_PROCS)

MODEL_HUB = config["config_model"]["MODEL_HUB"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = " , device)

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True


job_nb = sys.argv[1]

# Create a directory for storing results
results_dir = f"./results_evaluation_sans_finetuning{job_nb}"  # Ensure correct naming
os.makedirs(results_dir, exist_ok=True)

# Save configuration to the correct directory
config_path = os.path.join(results_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_path}")

### Load model ###

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_HUB)

print("max_position_embeddings = " , model.config.max_position_embeddings) 

# Load dataset (e.g., CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')

def len_distrib(batch):

    len_articles = []
    len_highlights = []
    
    for article, highlight in zip(batch["article"], batch["highlights"]):
        len_articles.append(len(tokenizer(article, truncation=False)["input_ids"]))
        len_highlights.append(len(tokenizer(highlight, truncation=False)["input_ids"]))


    source = tokenizer(batch["article"],truncation=True, max_length=tokenizer.model_max_length)
    resume = tokenizer(batch["highlights"],truncation=True, max_length=tokenizer.model_max_length)

    return {
        'input_ids': source['input_ids'], 
        'input_mask': source['attention_mask'],
        'input_len': len_articles,
        'target_ids': resume['input_ids'], 
        'target_mask': resume['attention_mask'],
        'target_len': len_highlights
        }

dataset = dataset.map(len_distrib,num_proc=NUM_PROCS,batched=True,batch_size=BATCH_SIZE)# Save the Hugging Face dataset

# Define the custom collate function
def collate_fn(batch):
    """
    Custom collate function that add padding for each batch.
    """

    id = [item['id'] for item in batch]

    # Pad the tokenized content
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    attention_mask = [torch.tensor(item['input_mask'], dtype=torch.long) for item in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # decoder_input_ids  = [torch.tensor(item['target_ids'][:-1], dtype=torch.long) for item in batch]
    # decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)     
    
    # decoder_attention_mask = [torch.tensor(item['target_mask'][:-1], dtype=torch.long) for item in batch]
    # decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0)
    
    input_len = torch.tensor([item['input_len'] for item in batch], dtype=torch.long)

    target_len = torch.tensor([item['target_len'] for item in batch], dtype=torch.long)

    # # Labels should be the same as decoder_input_ids (BART-style training)
    # labels = [torch.tensor(item['target_ids'][1:], dtype=torch.long) for item in batch]
    # labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)  
    # labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss computation

    highlights = [item['highlights'] for item in batch]

    return {
        'id':id,
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        # 'decoder_input_ids':decoder_input_ids,
        # 'decoder_attention_mask':decoder_attention_mask,
        # 'labels': labels,
        'input_len': input_len,
        'target_len': target_len,
        'highlights':highlights
    }


params =  {
    'batch_size': config["config_training"]["VALID_BATCH_SIZE"],
    'shuffle': False,
    'collate_fn':collate_fn,
    'num_workers': NUM_LOADER,
    'pin_memory': True  #  Enables faster GPU transfers
    }

# Return batch of data
loader = DataLoader(dataset, **params)

# Return acurate score
rouge = evaluate.load('rouge')


with open(os.path.join(results_dir, "rouge.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["rouge1", "rouge2", "rougeL"]
    writer.writerow(field)

with open(os.path.join(results_dir, "len.csv"), 'w', newline='') as file:
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
              **config['config_generate']
              )   

        mask = ~torch.isin(generated_ids, exclude_ids) #mask to skip the special tokens 
        generate_len = mask.sum(dim=1)  

        with open(os.path.join(results_dir, "len.csv"), 'a', newline='') as file:
            writer = csv.writer(file)
    
            min_len = min(len(batch["id"]), len(generate_len))

            for i in range(min_len):
                try:
                    writer.writerow([batch["id"][i], batch["input_len"][i], batch["target_len"][i], generate_len[i].item()])
                except Exception as e:
                    print(f"Error writing row {i}: {e}")
                    
        # Compute ROUGE scores here
        generated_txt = [text for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        references = [ref for ref in batch["highlights"]]

        rouge_results = rouge.compute(predictions=generated_txt,
                                      references=references,
                                      use_stemmer=True  # Ensures correct ROUGE comparison
                                      )
        
        
        with open(os.path.join(results_dir, "rouge.csv"), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL']])

        rouge1_score += rouge_results['rouge1']* min_len
        rouge2_score += rouge_results['rouge2']* min_len
        rougeL_score += rouge_results['rougeL']* min_len

        nb_sample += min_len
       

with open(os.path.join(results_dir, "rouge_total.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["Total_rouge1", "Total_rouge2", "Total_rougeL"]
    writer.writerow(field)
    writer.writerow([rouge1_score/nb_sample, rouge2_score/nb_sample, rougeL_score/nb_sample])
