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
from repilot_bart.modeling_repilot_bart import shift_tokens_right


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

    # model = BartForConditionalGeneration.from_pretrained(_CHECKPOINT_FOR_DOC)
    model.to(device)


    ##DENOISING TASK
    print("##DENOISING TASK WITH INPUT_IDS")

    TXT = "My friends are <mask> but they eat too many carbs."

    input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

    decoder_input_ids = shift_tokens_right(input_ids, model.config.pad_token_id, model.config.decoder_start_token_id)
    logger.info(f"input_ids.shape {input_ids.shape}")
    logger.info(f"decoder_input_ids.shape {decoder_input_ids.shape}")

    output = model(input_ids=input_ids.to(device), decoder_input_ids=decoder_input_ids.to(device))
    logits = output.logits
    print("logits.shape : ",logits.shape)

    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)

    for key in predictions:
        print(tokenizer.decode([key]))



