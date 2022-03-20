import os
import itertools
from turtle import forward
from cv2 import polarToCart
import einops
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from torchvision.utils import save_image
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict, emb_dim=512, block_size=2000):
        super().__init__()
        self.args = args
        self.text_dict = text_dict

        # Load VQ-VAE and set all parameters to no grad
        from .pose_vqvae_vit_model_spl_seperate import PoseVitVQVAE
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  PoseVitVQVAE.load_from_checkpoint(args.pose_vqvae, hparams_file=args.hparams_file)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        self.src_embedding = nn.Embedding(len(text_dict), emb_dim, padding_idx=text_dict.pad())
        self.pad_idx = self.vqvae.args.n_codes
        self.bos_idx = self.vqvae.args.n_codes + 1
        self.eos_idx = self.vqvae.args.n_codes + 2
        self.tgt_embedding = nn.Embedding(self.vqvae.args.n_codes + 3, emb_dim, padding_idx=self.vqvae.args.n_codes)
        self.tem_pos_emb = nn.Parameter(torch.zeros(block_size, emb_dim))
        self.spa_pos_emb = nn.Parameter(torch.zeros(3, emb_dim))

        self.transformer = Transformer(emb_dim=512, depth=6, block_size=2000)
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        points_tokens, points_embedding, skel_len = self._points2tokens(batch) # [bs, sum(skel_len), 3]
        word_tokens = batch["gloss_id"]
        bs, src_len = word_tokens.size()

        src_feat = self.src_embedding(word_tokens) + self.tem_pos_emb[:src_len, :]
        src_mask = word_tokens.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        point_feat = self.tgt_embedding(points_tokens) # [bs, sum(skel_len), 3, emb_dim]
        point_feat_with_spatial = point_feat + self.spa_pos_emb  # [bs, sum(skel_len), 3, emb_dim]
        temporal_feat = point_feat_with_spatial.sum(-2) # [bs, sum(skel_len), emb_dim]

        max_len = max(skel_len)
        tgt_feat = []
        pad_feat = self.tgt_embedding.weight[self.pad_idx:self.pad_idx+1, :].unsqueeze_(1)
        bos_feat = self.tgt_embedding.weight[self.bos_idx:self.bos_idx+1, :].unsqueeze_(1)

        tgt_out = []
        eos_token = points_tokens.new(1, 1, 3).fill_(self.eos_idx)
        pad_token = points_tokens.new(1, 1, 3).fill_(self.pad_idx)
        for i in range(bs):
            cur_len = skel_len[i]
            cur_out = torch.cat([points_tokens[i:i+1, :cur_len], eos_token, pad_token.repeat(1, max_len - cur_len, 1)], dim=1)
            tgt_out.append(cur_out)

            cur_feat = temporal_feat[i:i+1, :cur_len, :]
            cur_feat = torch.cat([bos_feat, cur_feat, pad_feat.repeat(1, max_len - cur_len, 1)], dim=1)
            tgt_feat.append(cur_feat)
            
        tgt_out = torch.cat(tgt_out, dim=0)

        tgt_feat = torch.cat(tgt_feat, dim=0) + self.tem_pos_emb[:max_len+1, :]

        tgt_mask = self._get_mask(skel_len+1, max_len+1, tgt_feat.device).unsqueeze_(1).unsqueeze_(2)
        
        pred_emb = self.transformer(src_feat, src_mask, tgt_feat, tgt_mask) #[bs, max_len+1, emd_dim]
        
        pred_emb = pred_emb.unsqueeze(-2).repeat(1, 1, 3, 1)  + self.spa_pos_emb # [bs, max_len+1, 3, emd_dim]

        logits = torch.matmul(pred_emb, self.tgt_embedding.weight.t()) # [bs, max_len+1, 3, n_codes+3]
        
        logits = logits.view(-1, logits.size(-1))
        tgt_out = tgt_out.view(-1)
        tgt_no_pad = tgt_out.ne(self.pad_idx)
        loss = F.cross_entropy(logits, tgt_out, ignore_index=self.pad_idx, reduction="sum")
        loss = loss / tgt_no_pad.sum()

        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)


    def generate(self, batch, temperature = 1., default_batch_size = 1):
        self.vqvae.eval()
        word_tokens = batch["gloss_id"]
        bos_feat = self.tgt_embedding.weight[self.bos_idx:self.bos_idx+1, :].unsqueeze_(1)


    @torch.no_grad()
    def _points2tokens(self, batch):
        points = batch["skel_3d"]  # [bs, 150]
        skel_len = batch["skel_len"]
        b, t, v = points.size()
        
        points = torch.cat([points[:, :skel_len[i], :] for i in range(b)], dim=1)
        points = einops.rearrange(points, "b t v -> (b t) v")
        pose = points[:, :24]
        rhand = points[:, 24:24+63]
        lhand = points[:, 87:150]

        points_tokens, points_embedding, commitment_loss = self.vqvae.encode(pose, rhand, lhand)
        points_tokens = einops.rearrange(points_tokens, "(b t) n -> b t n", b=b) # [bs, sum(skel_len), 3]
        return points_tokens, points_embedding, skel_len


    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        return [optimizer]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pose_vqvae', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')

        # VideoGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_dropout', type=float, default=0.3)
        parser.add_argument('--pkeep', type=float, default=1.0)
        parser.add_argument('--block_size', type=int, default=10000)
        parser.add_argument('--label_cond', action='store_true')
        return parser



class Transformer(nn.Module):
    def __init__(self, emb_dim=512, depth=6, block_size=2000):
        super().__init__()
        casual_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("casual_mask", casual_mask.bool().view(1, 1, block_size, block_size))

        self.encoder = Encoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)
        self.decoder = Decoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)


    def forward(self, src_feat, src_mask, tgt_feat, tgt_mask): 
        enc_out = self.encoder(src_feat, src_mask)
        bs, t, _ = tgt_feat.size()
        casual_mask = self.casual_mask[:, :, :t, :t]
        pad_future_mask = casual_mask & tgt_mask
        dec_out = self.decoder(tgt_feat, pad_future_mask, enc_out, src_mask)
        return dec_out
        

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2  =nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x
        

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        self.self_attn = Attention(heads, dim)
        self.cross_attn = Attention(heads, dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.norm3  = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, enc_out, tgt_mask, src_mask):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, pad_future_mask, enc_out, src_mask):
        for layer in self.layers:
            x = layer(x, enc_out, pad_future_mask, src_mask)
        return x


class Attention(nn.Module):
    def __init__(self, num_heads, size):
        super(Attention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v,mask):
        batch_size = k.size(0)
        num_heads = self.num_heads

        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3)) # [bs, head, q_len, kv_len]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf")) 

        attention = self.softmax(scores)
        context = torch.matmul(attention, v)

        context = (context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size))
        output = self.output_layer(context)
        return output