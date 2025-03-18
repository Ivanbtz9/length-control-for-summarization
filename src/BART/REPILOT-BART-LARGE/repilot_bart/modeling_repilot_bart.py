"""PyTorch REPILOT_BART model."""

import math
import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BartForConditionalGeneration, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.WARNING, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler()  #log to console
                    ])

logger = logging.getLogger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"


class RepilotBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Define the reversed position embedding module
        self.embed_reverse_positions = self._create_reverse_position_embedding()
        logger.info("RepilotBartForConditionalGeneration initialized successfully.")
        

    def _create_reverse_position_embedding(self):
        """Creates sinusoidal reversed position embeddings."""
        d_model = self.config.d_model
        max_len = self.config.max_position_embeddings
        padding_idx = self.config.pad_token_id

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-2 * (torch.arange(0, d_model) // 2) / d_model * math.log(10000.0))

        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        embedding = nn.Embedding(num_embeddings=max_len,
                                              embedding_dim=d_model,
                                              padding_idx=padding_idx,
                                              _weight=pe,
                                              _freeze=True)
        return embedding

    def _reverse_position_embedding(self, 
                                    input_ids:torch.Tensor, 
                                    target_len:Optional[torch.Tensor]=None)->torch.Tensor:
        """Computes reversed position indices for the decoder inputs."""
        mask = ~torch.isin(input_ids,torch.tensor([self.config.pad_token_id]))

        reversed_position_input  = torch.ones(mask.shape) * mask 
        
        if target_len is None:
            reversed_position_input = torch.flip(torch.flip(reversed_position_input , dims=(1,)).cumsum(dim=1), dims=(1,))
        else:
            for k in range(input_ids.size(-1)):
                reversed_position_input[:,k] = F.relu(target_len -k)
                print(reversed_position_input[:,k])

        #Add a gaussian noise
        normal_round = torch.randn(reversed_position_input.shape) * mask
        reversed_position_input = torch.abs(torch.round(reversed_position_input  + normal_round)).to(torch.long) #add a gausian noise and converte to long


        return reversed_position_input

    def forward(self, 
                input_ids:Optional[torch.Tensor]=None, 
                attention_mask:Optional[torch.Tensor]=None, 
                decoder_input_ids:Optional[torch.Tensor]=None, 
                decoder_attention_mask:Optional[torch.Tensor]=None,
                target_len:Optional[torch.Tensor]=None,
                decoder_inputs_embeds:Optional[torch.Tensor]=None,
                labels:Optional[torch.Tensor]=None, 
                **kwargs):
        """Overrides forward to inject reversed position embeddings into the decoder."""

        if input_ids is not None:
            logger.debug(f"Forward called with input_ids shape: {input_ids.shape}")
            print(input_ids)
        else:
            logger.warning("Forward called with input_ids=None")

        if decoder_input_ids is not None:
            # Compute reversed position indices
            reversed_position_input = self._reverse_position_embedding(decoder_input_ids, target_len)
            logger.debug(f"decoder_input_ids shape: {decoder_input_ids.shape}")
            print(reversed_position_input)
            
            # Get reversed position embeddings
            reversed_position_embeddings = self.embed_reverse_positions(reversed_position_input)

            #Get position embeddings 
            #position_embeddings = self.model.decoder.embed_positions(decoder_input_ids)

            # Compute standard token embeddings
            decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids) + reversed_position_embeddings #+ position_embeddings
            logger.debug(f"decoder_inputs_embeds shape: {decoder_inputs_embeds.shape}")

        # Call the original BART forward function with modified decoder inputs
        outputs =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

        logger.info("Forward pass completed.")
        return outputs
    
if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)
    model = RepilotBartForConditionalGeneration.from_pretrained(_CHECKPOINT_FOR_DOC)

    ##SUMMARY TASK

    ARTICLE_TO_SUMMARIZE = "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions.The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=model.config.max_position_embeddings, truncation=True, return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(input_ids=inputs["input_ids"], num_beams = 2, max_length = 4, target_len=torch.tensor([8]))
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))

    ##DENOISING TASK

    # TXT = "My friends are <mask> but they eat too many carbs."
    # input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    # output = model(input_ids)
    # logits = output.logits

    # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # probs = logits[0, masked_index].softmax(dim=0)
    # values, predictions = probs.topk(5)

    # print(tokenizer.decode(predictions).split())




