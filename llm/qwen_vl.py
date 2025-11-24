from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel, Qwen2_5_VLModel, Qwen2_5_VisionTransformerPretrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import wandb


class QwenVLRSEmb(Qwen2_5_VLPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.pool_type = kwargs.pop("pool_type")
        print(f"pool_type is {self.pool_type}")
        self.tau = kwargs.pop("tau")
        self.seperate_eval = kwargs.pop("seperate_eval")
        self.concat_features = kwargs.pop("concat_features")
        self.enable_moelora = kwargs.pop("enable_moelora")
        self.info_nce = kwargs.pop("info_nce")
        self.multimodal_embed=kwargs.pop("multimodal_embed")
        self.gate_weights = []
        # if self.enable_moelora:
            # self.item_emb_path = kwargs.pop("item_emb_path")
            # self.init_rec_emb()
        self.tau = nn.Parameter(torch.FloatTensor([3]))
        self.post_init()
        
    def init_rec_emb(self):
        srs_item_emb = pickle.load(open(self.item_emb_path, "rb"))
        srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_gate_weights(self, item_ids):
        item_embeddings = self.get_item_emb(item_ids)
        gate_weights = self.router(item_embeddings)
        self.gate_weights = gate_weights
        return gate_weights

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        
    def is_double_batch(self):
        # true for now
        return True


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(input_ids.device)
            else:
                sequence_lengths = -1
        transformer_outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        # hidden_states = transformer_outputs[0]
        # print(f"transformer_outputs: {transformer_outputs}")
        # hidden_states = F.normalize(hidden_states, p=2, dim=-1) 
        # print(f"Hidden states shape: {hidden_states.shape}")
        # print(f"Hidden states  : {hidden_states}")
        hidden_states = transformer_outputs.last_hidden_state
        # print(f"last_hidden_state shape: {hidden_states.shape}")
        # print(f"Hidden last_hidden_state  : {hidden_states}")
        text_image_pooled = self._get_pool_emb(hidden_states, sequence_lengths=sequence_lengths, pooled_mask=attention_mask)
        
            
        if not self.training:
            if self.multimodal_embed and self.seperate_eval:
                image_pooled = text_image_pooled[::2]  # Take elements at even indices
                text_pooled = text_image_pooled[1::2]
                if self.concat_features:
                    text_image_pooled = torch.cat([image_pooled, text_pooled], dim=1)
                else:
                    text_image_pooled = (image_pooled + text_pooled) / 2
            return SequenceClassifierOutputWithPast(
                loss=None,
                logits=text_image_pooled,  # 仅返回 text & image 结果
                past_key_values=None,
                hidden_states=text_image_pooled,
                attentions=None,
            )
        # print(f"text_image_pooled: {text_image_pooled.shape}")
        #  image_pooled could be random drop attribute 
        image_pooled = self.pool_embedding(text_image_pooled[::2])
        text_pooled = self.pool_embedding(text_image_pooled[1::2])
        # print(f"image_pooled: {image_pooled.shape}, text_pooled: {text_pooled.shape}")
        
        # print(F"image_pooled: {image_pooled}, text_pooled: {text_pooled}")
        # Contrastive loss computation
        loss = None
        # print(f"image_pooled: {image_pooled.shape}, text_pooled: {text_pooled.shape}")
        # print(f"image_pooled: {image_pooled}, text_pooled: {text_pooled}")
        # Apply the projectors
        # image_projected = self.image_projector(image_pooled)
        # text_projected = self.text_projector(text_pooled)
        if self.info_nce:
            loss_fct = InfoNCE_Loss(self.tau)
        else:
            loss_fct = Contrastive_Loss(self.tau)
        # if not self.enable_moelora:
        # return self.analyze_gradient(transformer_outputs, text_image_pooled, loss_fct, image_pooled, text_pooled)
            
        loss = loss_fct(image_pooled, text_pooled)
        


        if not return_dict:
            return (loss, image_pooled, text_pooled)
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=text_image_pooled,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    
    def pool_embedding(self, embedding):
        """
        将 [batch_size, 3584] 降维到 [batch_size, 768]
        前256个维度：每4个元素平均 (1024 -> 256)
        后512个维度：每5个元素平均 (2560 -> 512)
        """
        batch_size, original_dim = embedding.shape
        
        # 分割embedding
        # 前1024个维度 -> 256维 (每4个平均)
        first_part = embedding[:, :1024]  # [batch_size, 1024]
        first_pooled = first_part.view(batch_size, 256, 4).mean(dim=2)  # [batch_size, 256]
        
        # 接下来2560个维度 -> 512维 (每5个平均)
        second_part = embedding[:, 1024:3584]  # [batch_size, 2560]
        second_pooled = second_part.view(batch_size, 512, 5).mean(dim=2)  # [batch_size, 512]
        
        # 拼接结果
        pooled = torch.cat([first_pooled, second_pooled], dim=1)  # [batch_size, 768]
        
        return pooled
    
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
        # 归一化为单位向量
        X = F.normalize(X, p=2, dim=-1)
        Y = F.normalize(Y, p=2, dim=-1)
        logits = (X @ Y.T) / self.temperature  # shape: (B, B)
        labels = torch.arange(X.size(0), device=X.device)  # shape: (B,)
        loss_X = F.cross_entropy(logits, labels, reduction='none')  # X as query
        loss_Y = F.cross_entropy(logits.T, labels, reduction='none')  # Y as query
        loss = (loss_X + loss_Y) / 2.0
        return loss.mean()

