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

    ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    theorical_bart_result = 'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
    target_len = torch.tensor([len(tokenizer(theorical_bart_result ,max_length=1024)["input_ids"])],dtype=torch.long,device=device)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"].to(device), num_beams=2, min_length=0, max_length=23,early_stopping=True, target_len=target_len)

    
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])

    
    






