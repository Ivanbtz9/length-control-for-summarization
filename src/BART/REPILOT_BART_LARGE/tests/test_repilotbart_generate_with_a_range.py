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

    # Phrase à résumer
    input_text = "Bart is a transformer model developed by Facebook AI and I try to make a wrapper that add reverse positional embeddings."
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # Paramètres
    max_summary_len = 5
    eos_token_id = model.config.eos_token_id
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)

    print("INPUT TEXT:", input_text)
    print("#"*60)

    # Encodage
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

    # Génération token par token
    for step in range(max_summary_len):
        with torch.no_grad():
            outputs = model(
                input_ids=None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                target_len=torch.tensor([3], device=device)
            )

            next_token_logits = outputs.logits[:, -1, :]  # logits du dernier token
             #temperature
            T = 1   
            ##gredy sampling
            #next_token_id = torch.argmax(next_token_logits, dim=-1)

            # Define a probability distribution
            probs = F.softmax(next_token_logits/T,dim=-1)
            categorical_dist = Categorical(probs)

            # Take a sample
            next_token_id = categorical_dist.sample()

            # Print the sampled index
            print(f"Sampled index: {next_token_id.item()}")

        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(1)], dim=-1)

        if next_token_id.item() == eos_token_id:
            print("EOS token generated, stopping.")
            break

    # Affichage
    decoded_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
    print(f"Step(s) {step +1}")
    print("→ Decoder Input IDs:", decoder_input_ids[0].tolist())
    print("→ Decoded output:", decoded_text)
    print("-"*50)