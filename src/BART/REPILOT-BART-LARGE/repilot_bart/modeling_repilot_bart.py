"""PyTorch REPILOT_BART model."""
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


from datasets import load_dataset, Dataset

from transformers import BartModel, AutoTokenizer
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
from transformers.generation import GenerationMixin
from transformers import BartConfig, BartPreTrainedModel 

from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def tokenize_and_len(batch,tokenizer)->dict:
    """
    Return tokenization of article and highlights with there len
    """

    len_articles = []
    len_highlights = []
    
    for article, highlight in zip(batch["article"], batch["highlights"]):
        len_articles.append(len(tokenizer(article, truncation=False)["input_ids"])-1) #Add -1 to skip the <bos> token 
        len_highlights.append(len(tokenizer(highlight, truncation=False)["input_ids"])-1) #Add -1 to skip the <bos> token 


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


BART_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BartConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)

class RepilotBartForConditionalGeneration(BartPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]


    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))) #not updated by backpropagation
        self.embed_reverse_positions = self._create_sinusoidal_position_embedding()
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # Initialize weights and apply final processing
        self.post_init()


    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


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
        return embedding

    def _get_reverse_position_decoder_ids(self, decoder_input_ids:torch.LongTensor, target_len:Optional[torch.Tensor]=None, gaussian_noise=False)->torch.LongTensor:
        """Computes reversed position indices for the decoder inputs."""
        
        mask = ~torch.isin(decoder_input_ids,torch.tensor([self.config.pad_token_id]))

        reversed_position_input  = torch.ones(mask.shape) * mask 
        
        if target_len is None:
            reversed_position_input = torch.flip(torch.flip(reversed_position_input , dims=(1,)).cumsum(dim=1), dims=(1,))
            logger.debug(f"reversed_position_input WITHOUT target_len{reversed_position_input.shape}")

        else:
            k = torch.arange(decoder_input_ids.size(-1), device=target_len.device)  # Create a tensor [0, 1, 2, ..., seq_len-1]
            reversed_position_input = F.relu(target_len.unsqueeze(1) - k)  # Broadcast subtraction over all positions
            logger.debug(f"reversed_position_input shape WITH target_len {reversed_position_input.shape}")
            print(reversed_position_input)

        if gaussian_noise:
            normal_round = torch.randn(reversed_position_input.shape) * mask
        else:
            normal_round = 0

        return torch.abs(torch.round(reversed_position_input  + normal_round)).to(torch.long)
    
    def _get_position_decoder_ids(self, decoder_input_ids:torch.LongTensor)->torch.LongTensor:
        """Computes position indices for the decoder inputs."""

        ignore_token_ids = torch.tensor([self.config.pad_token_id])
        
        mask = ~torch.isin(decoder_input_ids,ignore_token_ids)

        position_decoder_input_ids  = torch.ones(mask.shape) * mask 

        return position_decoder_input_ids.cumsum(dim=1).to(torch.long)


    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(BART_GENERATION_EXAMPLE)

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
        output_hidden_states: Optional[bool] = True,#it can be None 
        return_dict: Optional[bool] = None,
        target_len:Optional[torch.Tensor]=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                #A way to create decoder_input_ids with labels 
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if decoder_input_ids is not None:
                # Get reversed position indices
                reversed_position_input_ids = self._get_reverse_position_decoder_ids(decoder_input_ids, target_len)

                # Compute reversed position embeddings
                reversed_position_embeddings = self.embed_reverse_positions(reversed_position_input_ids)

                # Get position indices
                position_input_ids = self._get_position_decoder_ids(decoder_input_ids)

                # Compute position embeddings
                position_embeddings = self.model.decoder.embed_positions(position_input_ids)

                # Compute standard token embeddings
                decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids) + position_embeddings + reversed_position_embeddings 
                logger.debug(f"SUM decoder_inputs_embeds shape: {decoder_inputs_embeds.shape}")
                # decoder_input_ids = None
        else:
            logger.warning("Forward called with decoder_input_ids=None")


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

        # print(outputs)
        
        print("outputs.last_hidden_state.shape = ",outputs.last_hidden_state.shape)
        print(len(outputs.decoder_hidden_states))
        # print(outputs.decoder_hidden_states[0].shape)
        # print("outputs.encoder_last_hidden_state.shape = ",outputs.encoder_last_hidden_state.shape)
        # print(len(outputs.encoder_hidden_states))
        # print("outputs.encoder_last_hidden_state :", outputs.encoder_last_hidden_state.shape)
        # print("outputs.past_key_values :" ,outputs.past_key_values) # (batch_size, num_heads, seq_len, head_dim)
        # print("outputs.decoder_last_hidden_state.shape :" ,outputs.decoder_last_hidden_state)

        # sys.exit()
        

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        target_len=None,
        **kwargs,
        ):

        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]


        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "target_len":target_len,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past




if __name__ == "__main__":

        # Setup logging
    logging.basicConfig(level=logging.DEBUG, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.StreamHandler()  #log to console
                        ])

    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_FOR_DOC, clean_up_tokenization_spaces=True)
    model = RepilotBartForConditionalGeneration.from_pretrained(_CHECKPOINT_FOR_DOC)

    ##SUMMARY TASK
    batch = {"article":["Fine-tuning BART on the CNN/DailyMail dataset involves several steps","I want to reproduce the finetuning of the model Bart on the dataset dailymail."],
            "highlights":["Fine-tuning BART ","finetuning Bart on the dataset dailymail."],
            } 

    batch = tokenize_and_len(batch,tokenizer)

    # Pad the tokenized content
    input_ids = [torch.tensor(item, dtype=torch.long) for item in batch['input_ids']]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    attention_mask = [torch.tensor(item, dtype=torch.long) for item in batch['input_mask']]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    decoder_input_ids  = [torch.tensor(item[:-1], dtype=torch.long) for item in batch['target_ids']]
    decoder_input_ids = pad_sequence(decoder_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)     

    decoder_attention_mask = [torch.tensor(item[:-1], dtype=torch.long) for item in batch['target_mask']]
    decoder_attention_mask = pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0)

    target_len = torch.tensor([item for item in batch['target_len']], dtype=torch.long)

    # Labels should be the same as decoder_input_ids (BART-style training)
    labels = [torch.tensor(item[1:], dtype=torch.long) for item in batch['target_ids']]
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)  
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss computation

    batch =  {
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        'decoder_input_ids':decoder_input_ids,
        'decoder_attention_mask':decoder_attention_mask,
        'labels': labels,
        'target_len': target_len
    }

    ##DENOISING TASK

    # TXT = "My friends are <mask> but they eat too many carbs."
    # input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    # TARGET_TXT = "My friends are <mask> but they eat too many carbs."
    # # Manually append the EOS token ID
    # eos_token_id = tokenizer.eos_token_id
    # decoder_input_ids = torch.cat([tokenizer([TARGET_TXT], return_tensors="pt",add_special_tokens=False)["input_ids"], torch.tensor([[eos_token_id]])], dim=1)
    # decoder_input_ids = shift_tokens_right(decoder_input_ids, model.config.pad_token_id, model.config.decoder_start_token_id)
    # print("input_ids.shape", input_ids)
    # print("decoder_input_ids.shape", decoder_input_ids)

    # output = model(input_ids=input_ids,decoder_input_ids=decoder_input_ids)
    # logits = output.logits
    # print("logits.shape : ",logits.shape)

    # masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    # probs = logits[0, masked_index].softmax(dim=0)
    # values, predictions = probs.topk(5)

    # for key in predictions:
    #     print(tokenizer.decode([key]))

    ##COMPUTE LOSS TASK
    
    # output = model(**batch)
    # print(batch['input_ids'].shape)
    # print(batch['decoder_input_ids'].shape)
    # print(output.logits.shape)


    ##GENERATE TASK
    summary_ids = model.generate(input_ids=batch["input_ids"], num_beams = 2, max_length = 20, early_stopping=True, target_len=torch.tensor([8,5]))
    print(summary_ids)
    print(tokenizer.batch_decode(summary_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))




































