"""PyTorch REPILOT_BART model."""
import copy
import math
import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    replace_return_docstrings,
)
from transformers import BartConfig, AutoTokenizer

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


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
        self.embed_reverse_positions = self._create_sinusoidal_position_embedding()
        logger.info("RepilotBartForConditionalGeneration initialized successfully.")
        

    def _create_sinusoidal_position_embedding(self)->torch.nn.Embedding:
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
        
        logger.info("Iitialized sinusoidal position embedding successfully.")
        return embedding

    def _get_reverse_position_decoder_ids(self, decoder_input_ids:torch.LongTensor, target_len:Optional[torch.Tensor]=None, gaussian_noise=True)->torch.LongTensor:
        """Computes reversed position indices for the decoder inputs."""
        
        mask = ~torch.isin(decoder_input_ids,torch.tensor([self.config.pad_token_id]))

        reversed_position_input  = torch.ones(mask.shape) * mask 
        
        if target_len is None:
            reversed_position_input = torch.flip(torch.flip(reversed_position_input , dims=(1,)).cumsum(dim=1), dims=(1,))
            logger.debug(f"Shape of reversed_position_input {reversed_position_input.shape}")

        else:
            for k in range(decoder_input_ids.size(-1)):
                reversed_position_input[:,k] = F.relu(target_len -k)
                logger.debug(f"reversed_position_input[:,{k}] {reversed_position_input[:,k]}")

        if gaussian_noise:
            normal_round = torch.randn(reversed_position_input.shape) * mask
        else:
            normal_round = 0

        return torch.abs(torch.round(reversed_position_input  + normal_round)).to(torch.long)
    
    def _get_position_decoder_ids(self, decoder_input_ids:torch.LongTensor)->torch.LongTensor:
        """Computes position indices for the decoder inputs."""
        
        mask = ~torch.isin(decoder_input_ids,torch.tensor([self.config.pad_token_id]))

        position_decoder_input_ids  = torch.ones(mask.shape) * mask 
        
        return position_decoder_input_ids.cumsum(dim=1).to(torch.long)
    

    
    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            target_len:Optional[torch.Tensor]=None
        )-> Union[Tuple, Seq2SeqLMOutput]:
        """Overrides forward to inject reversed position embeddings into the decoder."""


        if input_ids is not None:
            logger.info(f"Forward called with input_ids shape: {input_ids.shape}")
        else:
            logger.warning("Forward called with input_ids=None")

        if decoder_input_ids is not None:

            # Get reversed position indices
            reversed_position_input_ids = self._get_reverse_position_decoder_ids(decoder_input_ids, target_len)
            logger.info(f"reversed_position_input_ids shape: {reversed_position_input_ids.shape}")
            
            # Compute reversed position embeddings
            reversed_position_embeddings = self.embed_reverse_positions(reversed_position_input_ids)
            logger.info(f"reversed_position_embeddings shape: {reversed_position_embeddings.shape}")

            # Get position indices
            position_input_ids = self._get_position_decoder_ids(decoder_input_ids)
            logger.info(f"position_input_ids shape: {position_input_ids.shape}")
            
            # Compute position embeddings
            position_embeddings = self.model.decoder.embed_positions(position_input_ids)
            logger.info(f"position_embeddings shape: {position_embeddings.shape}")

            # Compute standard token embeddings
            decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids) + reversed_position_embeddings + position_embeddings
            logger.debug(f"decoder_inputs_embeds shape: {decoder_inputs_embeds.shape}")

        # Call the original BART forward function with modified decoder inputs
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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




