import os
import itertools
from turtle import forward
from cv2 import polarToCart
import einops
from matplotlib.pyplot import text
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
from util.plot_videos import draw_frame_2D
import torchvision
import cv2


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
        self.src_embedding.apply(self.init_bert_weights)
        self.pad_idx = self.vqvae.args.n_codes
        self.bos_idx = self.vqvae.args.n_codes + 1
        self.eos_idx = self.vqvae.args.n_codes + 2
        self.tgt_embedding = nn.Embedding(self.vqvae.args.n_codes + 3, emb_dim, padding_idx=self.vqvae.args.n_codes)
        self.tgt_embedding.apply(self.init_bert_weights)

        self.tem_pos_emb = nn.Parameter(torch.zeros(block_size, emb_dim))
        self.spa_pos_emb = nn.Parameter(torch.zeros(3, emb_dim))

        self.transformer = Transformer(emb_dim=512, depth=6, block_size=2000)
        self.out_linear = nn.Linear(emb_dim, emb_dim*3)
        self.transformer.apply(self.init_bert_weights)
        # backward
        self.conv1 = nn.Conv2d(256, 256, (5, 3), (1,3), (2,0))
        self.norm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(256, 256, 3, 1, 1)
        self.norm2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.ctc_out = nn.Linear(256, len(text_dict))

        self.ctcLoss = nn.CTCLoss(blank=text_dict.blank(), reduction="mean", zero_infinity=True)

        self.save_hyperparameters()

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def training_step(self, batch, batch_idx, mode="train"):
        self.vqvae.eval()
        points_tokens, points_embedding, skel_len = self._points2tokens(batch) # [sum(skel_len), 3]
        word_tokens = batch["gloss_id"]
        word_len = batch["gloss_len"]
        bs, src_len = word_tokens.size()

        src_feat = self.src_embedding(word_tokens) + self.tem_pos_emb[:src_len, :]
        src_mask = word_tokens.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        point_feat = self.tgt_embedding(points_tokens) # [sum(skel_len), 3, emb_dim]
        temporal_feat = point_feat + self.spa_pos_emb  # [sum(skel_len), 3, emb_dim]
        temporal_feat = temporal_feat.sum(-2) # [sum(skel_len), emb_dim]

        max_len = max(skel_len)
        
        pad_feat = self.tgt_embedding.weight[self.pad_idx:self.pad_idx+1, :].unsqueeze_(1)
        bos_feat = self.tgt_embedding.weight[self.bos_idx:self.bos_idx+1, :].unsqueeze_(1)
        
        eos_token = points_tokens.new(1, 1, 3).fill_(self.eos_idx)
        pad_token = points_tokens.new(1, 1, 3).fill_(self.pad_idx)
        
        tgt_out = []
        tgt_feat = []
        end_len = start_len = 0
        for i in range(bs):
            end_len += skel_len[i]
            start_len = end_len - skel_len[i]
            cur_out = points_tokens[start_len:end_len].unsqueeze(0)
            cur_out = torch.cat([cur_out, eos_token, pad_token.repeat(1, max_len - skel_len[i], 1)], dim=1)
            tgt_out.append(cur_out)

            cur_feat = temporal_feat[start_len:end_len, :].unsqueeze(0)
            cur_feat = torch.cat([bos_feat, cur_feat, pad_feat.repeat(1, max_len - skel_len[i], 1)], dim=1)
            tgt_feat.append(cur_feat)
        
        tgt_out = torch.cat(tgt_out, dim=0)  # [bs, max_len+1, 3]
        tgt_feat = torch.cat(tgt_feat, dim=0) + self.tem_pos_emb[:max_len+1, :] # [bs, max_len+1, emb_dim]

        tgt_mask = self._get_mask(skel_len+1, max_len+1, tgt_feat.device).unsqueeze_(1).unsqueeze_(2)
        
        pred_emb = self.transformer(src_feat, src_mask, tgt_feat, tgt_mask) # [bs, max_len+1, emd_dim]
        # pred_emb = pred_emb.unsqueeze(-2).repeat(1, 1, 3, 1)  + self.spa_pos_emb # [bs, max_len+1, 3, emd_dim]
        pred_emb = self.out_linear(pred_emb) # [bs, max_len+1, 3*emd_dim]
        pred_emb = einops.rearrange(pred_emb, "b t (n h) -> b t n h", n=3)
        logits_out = torch.matmul(pred_emb, self.tgt_embedding.weight.t()) # [bs, max_len+1, 3, n_codes+3]

        logits = logits_out.view(-1, logits_out.size(-1))
        tgt_out = tgt_out.view(-1)
        tgt_no_pad = tgt_out.ne(self.pad_idx)
        ce_loss = F.cross_entropy(logits, tgt_out, ignore_index=self.pad_idx, reduction="sum")
        ce_loss = ce_loss / tgt_no_pad.sum()
        self.log('{}/ce_loss'.format(mode), ce_loss.detach(), prog_bar=True)
    
        back_logits = logits_out.clone()
        back_logits = back_logits[:, :-1, :, :-3]   # [bs, max_len, 3, n_codes]
        predicts = F.gumbel_softmax(back_logits, tau=0.2, hard=True) # [bs, max_len+1, 3, n_codes]
        # predicts_feat = torch.matmul(predicts, self.tgt_embedding.weight) # [bs, max_len+1, 3, emb_dim]
        predicts = torch.matmul(predicts, self.vqvae.codebook.embeddings)  # [bs, max_len, 3, emb_dim]

        
        # recons loss
        rec_predicts = []
        points = batch["skel_3d"] # [bs, max_len, 150]
        ori_points = []       
        for i in range(bs):
            cur_points = points[i, :skel_len[i], :] 
            ori_points.append(cur_points)
            cur_pred = predicts[i, :skel_len[i], :, :]
            rec_predicts.append(cur_pred)
        ori_points = torch.cat(ori_points, dim=0) # [sum(skel_len), 150]
        rec_predicts = torch.cat(rec_predicts, dim=0) # [sum(skel_len), 3, emb_dim]
        
        
        rec_predicts = einops.rearrange(rec_predicts, "b n h -> b h n")
        dec_pose, dec_rhand, dec_lhand = self.vqvae.decode(rec_predicts)

        pose = ori_points[:, :24]
        rhand = ori_points[:, 24:24+63]
        lhand = ori_points[:, 87:150]
        pose_rec_loss = torch.abs(pose - dec_pose).mean()
        rhand_rec_loss = torch.abs(rhand - dec_rhand).mean()
        lhand_rec_loss = torch.abs(lhand - dec_lhand).mean()

        rec_loss = pose_rec_loss + rhand_rec_loss + lhand_rec_loss
        self.log('{}/pose_rec_loss'.format(mode), pose_rec_loss.detach(), prog_bar=True)
        self.log('{}/rhand_rec_loss'.format(mode), rhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/lhand_rec_loss'.format(mode), lhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/rec_loss'.format(mode), rec_loss.detach(), prog_bar=True)


        # ctc loss
        x = einops.rearrange(predicts, "b t n h -> b h t n")
        x = self.conv1(x).squeeze(-1)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = einops.rearrange(x, "b h t -> b t h")
        ctc_logits = self.ctc_out(x)

        lprobs = ctc_logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        
        ctc_loss = self.ctcLoss(lprobs.cpu(), word_tokens.cpu(), (skel_len // 4).cpu(), word_len.cpu()).to(lprobs.device)
        self.log('{}/ctc_loss'.format(mode), ctc_loss.detach(), prog_bar=True)

        loss = ce_loss + rec_loss + ctc_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)

        if mode == "train" and self.global_step % 200 == 0:
            self.vis(pose, rhand, lhand, mode, "ori_vis")
            self.vis(dec_pose, dec_rhand, dec_lhand, mode, "dec_vis")
            
            orig_tokens = points_tokens[:skel_len[0], :].view(-1) # [1_len, 3]
            pred_tokens = torch.argmax(logits_out[0, :skel_len[0], :], dim=-1).view(-1) # [1_len, 3]

            self.vis_token(pred_tokens, "recon")
            self.vis_token(orig_tokens, "origin")
        return loss


    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, "val")
        # self.generate(batch)

    @torch.no_grad()
    def generate(self, batch, batch_idx, total_seq_len=200, temperature = 1., default_batch_size = 1, ):
        self.vqvae.eval()
        word_tokens = batch["gloss_id"]
        bs, src_len = word_tokens.size()
        
        src_feat = self.src_embedding(word_tokens) + self.tem_pos_emb[:src_len, :]
        src_mask = word_tokens.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        bos_feat = self.tgt_embedding.weight[self.bos_idx:self.bos_idx+1, :].unsqueeze_(1) # [1, 1, emb_dim]
        tgt_feat = bos_feat
        tgt_mask = None
        res_tokens = []
        res_logits = []
        for _ in range(total_seq_len):
            pred_emb = self.transformer(src_feat, src_mask, tgt_feat, tgt_mask) # [1, t, emb_dim]
            pred_emb = self.out_linear(pred_emb)[:, -1:, :] # [1, 1, 3*emb_dim]
            pred_emb = einops.rearrange(pred_emb, "b t (n h) -> b t n h", n=3) # [1, 1, 3, emb_dim]
            logits = torch.matmul(pred_emb, self.tgt_embedding.weight.t()) # [1, 1, 3, vocab_size]
            res_logits.append(logits)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            # sampled = torch.argmax(logits, dim=-1) # [1, 1, 3]
            res_tokens.append(sampled)
            if (sampled == self.eos_idx).any():
                break
            sampled_feat = self.tgt_embedding(sampled) # [1, 1, 3]
            sampled_feat = sampled_feat + self.spa_pos_emb  # [1, 1, 3, emb_dim]
            sampled_feat = sampled_feat.sum(-2) # [1, 1, emb_dim]
            tgt_feat = torch.cat([tgt_feat, sampled_feat], dim=-2)  # [1, t+1, emb_dim]
        res_tokens =  torch.cat(res_tokens, dim=1).view(-1) # [1, t, 3]
        # self.vis_token(res_tokens, "pred")
        # print("res_tokens: ", res_tokens.shape, res_tokens)

        res_logits = torch.cat(res_logits, dim=1)[..., :-3].contiguous() # [1,t, 3, vocab_size]
        # print("res_logits: ", res_logits.shape)
        predicts = F.gumbel_softmax(res_logits, tau=0.1, hard=True) # [1, t, 3, n_codes]
        predicts = torch.matmul(predicts, self.vqvae.codebook.embeddings)  # [1, t, 3, emb_dim]
        predicts = einops.rearrange(predicts, " b t n h -> (b t) h n") 
        pred_pose, pred_rhand, pred_lhand = self.vqvae.decode(predicts)
        pred_points = torch.cat([pred_pose, pred_rhand, pred_lhand], dim=-1).detach().cpu().numpy() # [t, 150]
        show_img = []
        for j in range(pred_points.shape[0]):
            frame_joints = pred_points[j]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d)
            show_img.append(im)
        show_img = np.concatenate(show_img, axis=1) # [h, w*16, c]
        cv2.imwrite("Data/predictions/show_gumbel_{}.png".format(batch_idx), show_img)
        # points = batch["skel_3d"].squeeze(0) # [1, max_len, 150]
        # print("points: ", points.shape)
        # exit()
        return res_tokens
        

    def vis_token(self, pred_tokens, name):
        pred_tokens = " ".join([str(x) for x in pred_tokens.cpu().numpy().tolist()])
        self.logger.experiment.add_text("{}_points".format(name), pred_tokens, self.current_epoch)

    def vis(self, pose, rhand, lhand, mode, name):
        points = torch.cat([pose, rhand, lhand], dim=-1).detach().cpu().numpy()
        # points: [bs, 150]
        show_img = []
        for j in range(16):
            frame_joints = points[j]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d)
            show_img.append(im)
        show_img = np.concatenate(show_img, axis=1) # [h, w*16, c]
        show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
        show_img = torchvision.utils.make_grid(show_img, )
        self.logger.experiment.add_image("{}/{}".format(mode, name), show_img, self.global_step)

    


    @torch.no_grad()
    def _points2tokens(self, batch):
        points = batch["skel_3d"]  # [bs, t, 150]
        skel_len = batch["skel_len"]
        b, t, v = points.size()

        points = torch.cat([points[i, :skel_len[i], :] for i in range(b)], dim=0) # [sum(skel_len), 150]
       
        pose = points[:, :24]
        rhand = points[:, 24:24+63]
        lhand = points[:, 87:150]

        points_tokens, points_embedding, commitment_loss = self.vqvae.encode(pose, rhand, lhand) 
        return points_tokens, points_embedding, skel_len # [sum(skel_len), 3], # [sum(skel_len), emb_dim, 3]


    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, betas=(0.9, 0.999))
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

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


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
        if tgt_mask is not None:
            pad_future_mask = casual_mask & tgt_mask
        else:
            pad_future_mask = casual_mask
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
        self.norm1 = BertLayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = BertLayerNorm(dim)
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

        self.norm1 = BertLayerNorm(dim)
        self.norm2  = BertLayerNorm(dim)
        self.norm3  = BertLayerNorm(dim)

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