import copy
import math
import logging
import sys, os
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import cuda
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from transformers import AutoTokenizer, BartForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from repilot_bart.modeling_repilot_bart import RepilotBartForConditionalGeneration
from repilot_bart.modeling_repilot_bart import shift_tokens_right, tokenize_and_len


_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"

# Setup logging
logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler()  #log to console
                    ])

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)
    model = RepilotBartForConditionalGeneration.from_pretrained(_CHECKPOINT_FOR_DOC)
    # model.gaussian_noise = False
    # model.repilot_status = False

    model.to(device)

    batch = {"article":["Fine-tuning BART on the CNN/DailyMail dataset involves several steps","I want to reproduce the finetuning of the model Bart on the dataset dailymail."],
            "highlights":["Fine-tuning BART ","finetuning Bart on the dataset dailymail."],
            } 

    batch = tokenize_and_len(batch,tokenizer)

    # Pad the tokenized content
    input_ids = [torch.tensor(item, dtype=torch.long) for item in batch['input_ids']]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask = [torch.tensor(item, dtype=torch.long) for item in batch['input_mask']]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    decoder_input_ids  = [shift_tokens_right(torch.tensor(item, dtype=torch.long).unsqueeze(0), model.config.pad_token_id, model.config.decoder_start_token_id).squeeze(0) for item in batch['target_ids']]
    decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)    

    decoder_attention_mask = [torch.tensor(item, dtype=torch.long) for item in batch['target_mask']]
    decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0)

    target_len = torch.tensor([item for item in batch['target_len']], dtype=torch.long)

    # Labels should be the same as decoder_input_ids (BART-style training)
    labels = [torch.tensor(item, dtype=torch.long) for item in batch['target_ids']]
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)  
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss computation

    batch =  {
        'input_ids':input_ids.to(device),
        'attention_mask':attention_mask.to(device),
        'decoder_input_ids':decoder_input_ids.to(device),
        'decoder_attention_mask':decoder_attention_mask.to(device),
        'labels': labels.to(device),
        'target_len': target_len.to(device)
    }

    ##COMPUTE LOSS TASK
    print("##COMPUTE LOSS TASK")
    # print(batch['input_ids'])
    # print(batch['decoder_input_ids'])
    # print(batch['labels'])

    logger.debug(f"batch['input_ids'].shape {batch['input_ids'].shape}")
    logger.debug(f"batch['decoder_input_ids'].shape {batch['decoder_input_ids'].shape}")
    logger.debug(f"batch['labels'].shape {batch['labels'].shape}")

    output = model.forward(**batch)

    print("output.loss.item() = ",output.loss.item())

    ##GENERATE TASK
    print("##GENERATE TASK")
    summary_ids = model.generate(input_ids=batch["input_ids"], num_beams = 2, max_length = 20,do_sample=True, early_stopping=True, target_len=batch["target_len"] )
    print(summary_ids)
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    






