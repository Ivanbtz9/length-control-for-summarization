### MODULES ###
import copy
import math
import logging
import sys,os
import tqdm
import csv
from datetime import datetime 
import numpy as np
import pandas as pd
import json

from datasets import load_dataset

import torch
from torch import cuda
from torch import nn
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset

from transformers import AutoTokenizer, BartForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from repilot_bart.modeling_repilot_bart import RepilotBartForConditionalGeneration
from repilot_bart.modeling_repilot_bart import shift_tokens_right, tokenize_and_len


### CONFIG ###

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler()  #log to console
                    ])
logger = logging.getLogger(__name__)

time = datetime.now().strftime("%Y-%m-%d_%Hh%M")
job_nb = sys.argv[1]
path_to_save_checkpoints = sys.argv[2]

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"

NUM_PROCS = os.cpu_count()
logger.info(f"NUM_PROCS = {NUM_PROCS}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
logger.info(f"device available = {device}")


# Load configuration from JSON
with open('./config_finetune_repilot_bart_large.json', 'r') as f:
    config = json.load(f)
    print(config)
    print(type(config))

# Create a directory for storing results
results_dir = f"./config_and_code/finetuning_BART_large-{time}-{job_nb}"  # Ensure correct naming
os.makedirs(results_dir, exist_ok=True)

# Save configuration to the correct directory
config_path = os.path.join(results_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)


# Define source and destination paths
save_code_path = os.path.join(results_dir, os.path.basename(sys.argv[0]))
current_code_path = os.path.join(os.getcwd(), sys.argv[0])

try:
    # Copy the current script to results_dir
    shutil.copy(current_code_path, save_code_path)
    print(f"Successfully saved a copy of the script to: {save_code_path}")
except Exception as e:
    print(f"Error copying the script: {e}")


print(f"Configuration saved to {config_path}")

SEED = config['config_machine']["SEED"]
NUM_LOADER = config['config_machine']["NUM_LOADER"] #depends of the number of thread 


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

### LOAD DATA ###

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Check the dataset structure
print(dataset)


### LOAD MODEL ###

MODEL_HUB = config["config_model"]["MODEL_HUB"]
# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB, clean_up_tokenization_spaces=True)
model = BartForConditionalGeneration.from_pretrained(MODEL_HUB, forced_bos_token_id=tokenizer.bos_token_id)
print(tokenizer.model_max_length)
print(type(tokenizer))
print(type(model))


### DATA PROCESSING ###

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


dataset = dataset.map(len_distrib,num_proc=NUM_PROCS,batched=True,batch_size=config["config_machine"]["BATCH_SIZE"])# Save the Hugging Face dataset



# Define the custom collate function
def collate_fn(batch):
    """
    Custom collate function that add padding for each batch.
    """

    # Pad the tokenized content
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    attention_mask = [torch.tensor(item['input_mask'], dtype=torch.long) for item in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    decoder_input_ids  = [torch.tensor(item['target_ids'][:-1], dtype=torch.long) for item in batch]
    decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)     
    
    decoder_attention_mask = [torch.tensor(item['target_mask'][:-1], dtype=torch.long) for item in batch]
    decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0)
    
    # input_len = torch.tensor([item['input_len'] for item in batch], dtype=torch.long)

    # target_len = torch.tensor([item['target_len'] for item in batch], dtype=torch.long)

    # Labels should be the same as decoder_input_ids (BART-style training)
    labels = [torch.tensor(item['target_ids'][1:], dtype=torch.long) for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)  
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss computation

    return {
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        'decoder_input_ids':decoder_input_ids,
        'decoder_attention_mask':decoder_attention_mask,
        'labels': labels,
        # 'input_len': input_len,
        # 'target_len': target_len
    }


train_params = {
    'batch_size': config["config_training"]["TRAIN_BATCH_SIZE"],
    'shuffle': True,
    'collate_fn':collate_fn,
    'num_workers': NUM_LOADER,
    'pin_memory': True  #  Enables faster GPU transfers
    }

eval_params = {
    'batch_size': config["config_training"]["VALID_BATCH_SIZE"],
    'shuffle': False,
    'collate_fn':collate_fn,
    'num_workers': NUM_LOADER,
    'pin_memory': True  #  Enables faster GPU transfers
    }


# This will be used down for training and validation stage for the model.
train_loader = DataLoader(dataset["train"], **train_params)
eval_loader = DataLoader(dataset["validation"], **eval_params)

### TRAIN AND EVALUATION ###

def train(model, device, loader, optimizer, epoch, writer):
    """
    Function to call for training with the parameters passed from main function
    
    """

    total_loss = 0.0

    model.train()

    for _, batch in tqdm.tqdm(enumerate(loader, 0),desc=f'Total iterations: {len(loader)}', unit=" it"):

        batch = {key: val.to(device) for key, val in batch.items()}

        outputs = model(**batch)

        loss = outputs[0]
        total_loss += loss.item()
        
        writer.add_scalar(f"batch_loss/train", loss/len(batch["input_ids"]), epoch*len(loader) + _ )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {"loss": total_loss/len(loader)} 

def eval(model, device, loader, epoch, writer): #epoch,writer 

    """
    Function to be called for evaluate with the parameters passed from main function

    """

    total_loss = 0.0

    model.eval()
    
    with torch.no_grad():
        for _, batch in tqdm.tqdm(enumerate(loader, 0),desc=f'Total iterations: {len(loader)}', unit=" it"):

            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)

            loss = outputs[0]
            total_loss += loss.item()
            
            writer.add_scalar(f"batch_loss/evaluation", loss/len(batch["input_ids"]), epoch*len(loader) + _ )

    return {"loss": total_loss/len(loader)} 

def main(model, device, train_loader, eval_loader, optimizer, writer, scheduler, checkpoint_path, NB_EPOCHS, early_stopping_patience):
        
    best_loss = float('inf')
    patience_counter = 0
    current_lr = 0

    eval_loss, train_loss = [], []
    
    for epoch in range(NB_EPOCHS):
        outputs_train = train(model, device, train_loader, optimizer, epoch, writer)
        outputs_eval = eval(model, device, eval_loader, epoch, writer)
        
        train_loss.append(outputs_train["loss"])
        eval_loss.append(outputs_eval["loss"])

        writer.add_scalars("loss",{"train":outputs_train["loss"], "eval":outputs_eval["loss"] } , epoch)


        # Enregistrer le taux d'apprentissage actuel
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar("lr", current_lr, epoch)

        # Réduit automatiquement le taux d'apprentissage si la perte de validation ne diminue pas
        # sous les critères
        scheduler.step(eval_loss[-1])

        # Early Stopping
        if eval_loss[-1] < best_loss:
            best_loss = eval_loss[-1]
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            epoch_best = epoch+1
            print(f'Find a better model at the epoch {epoch+1} - Train Loss: {train_loss[-1]:.4f}, eval Loss: {eval_loss[-1]:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                print("current_lr : ", current_lr)
                print("epoch_best : ", epoch_best)
                break

        writer.close()




checkpoint_path = os.path.join(path_to_save_checkpoints,f"finetune_Bart_large/Bart-large-{time}-{job_nb}")
os.makedirs(checkpoint_path,exist_ok=True)

log_dir = f"./logs/Bart-large-{time}-{job_nb}" 
os.makedirs(log_dir,exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

# Defining the optimizer that will be used to tune the weights of the network in the training session. 
optimizer = torch.optim.AdamW(params = model.parameters(), 
                              lr=config['config_training']["LEARNING_RATE"], 
                              weight_decay=config['config_training']["weight_decay"])
#link : https://medium.com/@benjybo7/10-pytorch-optimizers-you-must-know-c99cf3390899

early_stopping_patience = config['config_training']["early_stopping_patience"]
reduce_lr_patience = config['config_training']["reduce_lr_patience"]
reduce_lr_factor = config['config_training']["reduce_lr_factor"]

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=reduce_lr_patience, factor=reduce_lr_factor)

main(model.to(device), 
     device, 
     train_loader, 
     eval_loader, 
     optimizer, 
     writer,
     scheduler,
     checkpoint_path, 
     config['config_training']["NB_EPOCHS"], 
     early_stopping_patience)