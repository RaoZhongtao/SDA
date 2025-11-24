# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec


class SASRecPLUS(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        
        self.args = args
        if not self.args.cross_attn:
            self.hidden_size = args.hidden_size
            llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.insert(llm_item_emb, -1, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
            if args.freeze_emb:
                self.item_emb.weight.requires_grad = False
            else:
                self.item_emb.weight.requires_grad = True
            self.adapter = nn.Sequential(
                nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
            )
            self.filter_init_modules = ["item_emb"]

        else:
            self.hidden_size = args.hidden_size
            lvlm_text_emb = pickle.load(open(args.lvlm_text_emb_path, "rb"))
            lvlm_image_emb = pickle.load(open(args.lvlm_image_emb_path, "rb"))
            
            lvlm_text_emb = np.insert(lvlm_text_emb, 0, values=np.zeros((1, lvlm_text_emb.shape[1])), axis=0)
            lvlm_text_emb = np.insert(lvlm_text_emb, -1, values=np.zeros((1, lvlm_text_emb.shape[1])), axis=0)
            
            lvlm_image_emb = np.insert(lvlm_image_emb, 0, values=np.zeros((1, lvlm_image_emb.shape[1])), axis=0)
            lvlm_image_emb = np.insert(lvlm_image_emb, -1, values=np.zeros((1, lvlm_image_emb.shape[1])), axis=0)

            self.lvlm_text_emb = nn.Embedding.from_pretrained(torch.Tensor(lvlm_text_emb))
            self.lvlm_image_emb = nn.Embedding.from_pretrained(torch.Tensor(lvlm_image_emb))
            
            if args.freeze_emb:
                self.lvlm_text_emb.weight.requires_grad = False
                self.lvlm_image_emb.weight.requires_grad = False
            else:
                self.lvlm_text_emb.weight.requires_grad = True
                self.lvlm_image_emb.weight.requires_grad = True

            self.fusion_module = CrossModalFusion()

            self.filter_init_modules = ["lvlm_text_emb", "lvlm_image_emb"]
        
        

        
        self._init_weights()

    
    def _get_embedding(self, log_seqs):
        
        if not self.args.cross_attn:
            item_seq_emb = self.item_emb(log_seqs)
            item_seq_emb = self.adapter(item_seq_emb)
            return item_seq_emb
        else:
            item_image_emb = self.lvlm_image_emb(log_seqs)
            item_text_emb = self.lvlm_text_emb(log_seqs)
            fused_emb = self.fusion_module(item_image_emb, item_text_emb)
            # item_seq_emb = self.adapter(fused_emb)
            return fused_emb
    

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs = seqs * (self.hidden_size ** 0.5)
        seqs = seqs + self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    


class Bert4RecPLUS(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))
        # self.pad_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        item_seq_emb[log_seqs==self.mask_token] = self.mask_embedding
        # item_seq_emb[log_seqs==0] = self.pad_embedding

        return item_seq_emb
    

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    


class GRU4RecPLUS(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.insert(llm_item_emb, -1, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        return item_seq_emb
    

    def log2feats(self, log_seqs):
        '''Get the representation of given sequence'''
        seqs = self.item_emb(log_seqs)
        seqs = self.adapter(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    



class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=3584, output_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_to_image_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.image_to_text_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.fc_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, text_embed, image_embed):
        text_enhanced = self.text_to_image_attn(text_embed, image_embed, image_embed)
        image_enhanced = self.image_to_text_attn(image_embed, text_embed, text_embed)
        fused_embedding = torch.cat([text_enhanced, image_enhanced], dim=-1)
        return self.fc_fusion(fused_embedding)