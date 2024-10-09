
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Attention(nn.Module):
    def __init__(self, dropout, cfg):
        super().__init__()
        self.hidden_size = cfg.model.transformer_encoder.att_dim
        self.num_attention_heads = cfg.model.transformer_encoder.att_head
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    # 没有 W^O的MHA
    def forward(self, query_states, key_states, value_states, shortcut=None):
        batch, L, D = query_states.size()
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        if shortcut is None:
            context_layer += query_states # 跳跃连接
        else:
            context_layer += shortcut

        context_layer = self.layer_norm(context_layer)
        #####################################
        
        return context_layer, attention_scores

class MSCP(nn.Module):
    def __init__(self, attention, dropout, cfg):
        super().__init__()
        self.attn = attention
        att_dim = cfg.model.transformer_encoder.att_dim
        self.ffn = nn.Sequential(
            nn.Linear(att_dim, att_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(att_dim*2, att_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(att_dim)

    def forward(self, x, k, v, shortcut=None):
        x, attn_weight = self.attn(x, k, v, shortcut)
        x = self.ln(x + self.ffn(x))    
        return x, attn_weight

class CVIE(nn.Module):
    def __init__(self, dropout, cfg):
        super().__init__()
        self.mha1 = Attention(dropout, cfg)
        self.mha2 = Attention(dropout, cfg)
        att_dim = cfg.model.transformer_encoder.att_dim
        self.mlp = nn.Sequential( # C19-3/4   
            nn.Linear(att_dim * 2, att_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(att_dim, att_dim),
        )

    def forward(self, x, y):
        guide, attn_weight1 = self.mha1(y, x, x, shortcut=x)
        guide2 = self.mlp(torch.cat([x, guide], dim=-1)) 
        x, attn_weight2 = self.mha2(guide2, x, x) 
        return x, attn_weight1, attn_weight2
    
class ChangeDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.transformer_encoder.input_dim
        self.dim = cfg.model.transformer_encoder.dim
        self.feat_dim = cfg.model.transformer_encoder.feat_dim
        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.num_layers = cfg.model.transformer_encoder.att_layer
        
        self.lambda_ = cfg.model.transformer_encoder.lambda_
        print(self.lambda_)
        dropout = 0.1


        self.conv = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0),
            nn.Dropout(0.1),    
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.att_dim, kernel_size=1, padding=0), 
        )

        h = w = 14
        self.w_embedding = nn.Embedding(w, int(self.att_dim / 2))
        self.h_embedding = nn.Embedding(h, int(self.att_dim / 2))
        nn.init.normal_(self.w_embedding.weight, std=0.02)
        nn.init.normal_(self.h_embedding.weight, std=0.02)

        self.intra = CVIE(dropout, cfg)
        
        self.inter = nn.ModuleList([MSCP(Attention(dropout, cfg), dropout, cfg) for i in range(self.num_layers)])
        
        self.mlp = nn.Sequential(
            nn.Linear(self.att_dim, int(self.att_dim*1)),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(int(self.att_dim*1), self.att_dim),
            nn.Sigmoid()
        )  

        self.reverser = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.att_dim * 2),
            nn.GELU(),
            nn.Linear(self.att_dim * 2, self.att_dim),
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(self.att_dim),
            nn.Linear(self.att_dim, 512, bias=False)
        )

        self.projection2 = nn.Sequential(
            nn.LayerNorm(self.att_dim),
            nn.Linear(self.att_dim, 512, bias=False)
        )

        self.logit_scale = nn.Parameter(torch.tensor([0.]))    

    def forward(self, input_1, input_2, args=None):
        
        batch_size, C, H, W = input_1.size()

        input_1 = self.conv(input_1)  
        input_2 = self.conv(input_2)

        short_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1) 
        short_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1) 


        pos_w = torch.arange(W).cuda()
        pos_h = torch.arange(H).cuda()
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(W, 1, 1), embed_h.unsqueeze(1).repeat(1, H, 1)], dim=-1)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch, d_model, h, w)
        input_1 = input_1 + position_embedding
        input_2 = input_2 + position_embedding

        input_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1) # [B, N, D]
        input_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1)  

        shortcut = input_1
        input_1, attn_weight11, attn_weight12 = self.intra(input_1, input_2) 
        input_2, attn_weight21, attn_weight22 = self.intra(input_2, shortcut) 

        x_c = [input_2 - input_1]     
        for i, layer_module in enumerate(self.inter):
            shortcut = input_1 
            input_1, attn_weight1 = layer_module(input_1, input_2, input_2) 
            input_2, attn_weight2 = layer_module(input_2, shortcut, shortcut) 
            x_c.append(input_2 - input_1)

        x_c = torch.stack(x_c, dim=2) # [B, N, L, D]
        alpha = self.mlp(x_c) # [B, N, L, D]
        x = torch.sum(alpha * x_c, dim=2) # [B, N, D]

        if self.training:
            if self.lambda_ > 0:
                rec = self.reverser(torch.cat([short_1, x], dim=-1)) 
                rec = self.projection(rec.mean(dim=1))
                rec = rec / rec.norm(p=2, dim=-1, keepdim=True)
                tar = self.projection2(short_2.mean(dim=1)) 
                tar = tar / tar.norm(p=2, dim=-1, keepdim=True)
                logits = torch.matmul(rec, tar.t()) * self.logit_scale.exp() 
                loss2 = clip_loss(logits, bi=True) * self.lambda_
            else:
                loss2 = torch.tensor(0.)
                
            return x, loss2
        else:
            return x

class AddSpatialInfo(nn.Module):
    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        x = torch.linspace(-1, 1, h) # [h]
        y = torch.linspace(-1, 1, w) # [w]
        x = x.view(1, h).repeat(w, 1) # [w, h]
        y = y.view(w, 1).repeat(1, h) # [w, h]
        coord_map = torch.stack([x, y], dim=0).to(img_feat.device)
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor, bi) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    if bi:
        image_loss = contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    else:
        return caption_loss