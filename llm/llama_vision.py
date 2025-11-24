from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.mllama.modeling_mllama import MllamaPreTrainedModel, MllamaVisionModel, MllamaForCausalLM, _prepare_cross_attention_mask


class LlamaVisionRSEmb(MllamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        
        self.vision_model = MllamaVisionModel._from_config(config.vision_config)
        self.language_model = MllamaForCausalLM._from_config(config.text_config)

        self.pool_type = kwargs.pop("pool_type")
        print(f"pool_type is {self.pool_type}")
        self.tau = kwargs.pop("tau")
        self.seperate_eval = kwargs.pop("seperate_eval")
        self.concat_features = kwargs.pop("concat_features")
        self.enable_moelora = kwargs.pop("enable_moelora")
        self.info_nce = kwargs.pop("info_nce")
        self.multimodal_embed=kwargs.pop("multimodal_embed")
        self.gate_weights = []

        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )

        self.tau = nn.Parameter(torch.FloatTensor([3]))
        self.post_init()
    
    def get_tau(self):
        return self.tau
    
    def init_rec_emb(self):
        srs_item_emb = pickle.load(open(self.item_emb_path, "rb"))
        srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_gate_weights(self, item_ids):
        item_embeddings = self.get_item_emb(item_ids)
        gate_weights = self.router(item_embeddings)
        self.gate_weights = gate_weights
        return gate_weights

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
        
    def is_double_batch(self):
        # true for now
        return True


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]



        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(input_ids.device)
            else:
                sequence_lengths = -1
        
        transformer_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        hidden_states = transformer_outputs.hidden_states[-1]
        
        # mask_sum = torch.sum(attention_mask, dim=1)
        # print("debugging llamaV  Any zero in pooled_mask sum:", (mask_sum == 0).any().item())
        if torch.isnan(hidden_states).any().item():
            print("debugging llamaV  hidden_states has nan:", torch.isnan(hidden_states).any().item())

        logits_pooled = self._get_pool_emb(hidden_states, sequence_lengths=sequence_lengths, pooled_mask=attention_mask)
        
            
        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=logits_pooled,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=logits_pooled, #无奈 不知道为啥这样写
            attentions=transformer_outputs.attentions,
        )
    

    def _get_pool_emb(self, hidden_states, sequence_lengths, pooled_mask):
        """get the logits according to pool type"""
        if self.pool_type == "last":    # take out the last token as LLM embedding
            pooled_emb = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                                       sequence_lengths]
        # average pooling all tokens as LLM embedding or average pooling attribute tokens as LLM embedding
        elif self.pool_type == "avg":   
            pooled_emb = torch.sum(hidden_states * pooled_mask.unsqueeze(-1), dim=1) / torch.sum(pooled_mask, dim=1).unsqueeze(-1) #sequence_lengths.unsqueeze(-1)

        return pooled_emb
    
    def get_item_emb(self, item_ids):
        item_embeddings = self.srs_emb(item_ids)
        
        if self.is_double_batch():
            item_embeddings = torch.repeat_interleave(item_embeddings, repeats=2, dim=0)
        
        return item_embeddings
    
    def analyze_gradient(self, transformer_outputs, text_image_pooled, loss_fct, image_pooled, text_pooled):
        loss = loss_fct(image_pooled, text_pooled.detach())
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=text_image_pooled,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        

class InfoNCE_Loss(nn.Module):
    def __init__(self, tau=1.0) -> None:
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        # Compute similarity logits
        logits = (X @ Y.T) / self.temperature  # shape: (B, B)

        # Labels: correct match is on the diagonal (i == j)
        labels = torch.arange(X.size(0), device=X.device)  # shape: (B,)

        # Cross entropy loss: X -> Y and Y -> X
        loss_X = F.cross_entropy(logits, labels, reduction='none')  # X as query
        loss_Y = F.cross_entropy(logits.T, labels, reduction='none')  # Y as query

        loss = (loss_X + loss_Y) / 2.0
        return loss.mean()

