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
from modules.mask_strategy import *


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()
        self.args = args
        self.text_dict = text_dict

        # Load VQ-VAE and set all parameters to no grad
        from .text2pose_model_ctc_pretrain_nat_freeze_emb import Text2PoseModel

        # if not os.path.exists(args.pose_vqvae):
        #     print("{} is not existed!".format(args.pose_vqvae))
        #     self.teacher = Text2PoseModel(args, text_dict)
        # else:
        #     print("load teacher model from {}".format(args.pose_vqvae))
        #     self.teacher =  Text2PoseModel.load_from_checkpoint(args.pose_vqvae, hparams_file=args.vqvae_hparams_file)
        #     for p in self.teacher.parameters():
        #         p.requires_grad = False
        #     self.teacher.vqvae.codebook._need_init = False
        #     self.teacher.eval()

        self.teacher = Text2PoseModel(args, text_dict)
        self.student = Text2PoseModel(args, text_dict)

        self.mask_idx = self.teacher.mask_idx
        self.pad_idx = self.teacher.pad_idx

        self.random = np.random.RandomState(1234)
        
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
        points_tokens, points_embedding, skel_len = self.teacher._points2tokens(batch) # [sum(skel_len), 3]
        
        word_tokens = batch["gloss_id"]
        word_len = batch["gloss_len"]
        bs, src_len = word_tokens.size()

        src_feat = self.teacher.src_embedding(word_tokens) + self.teacher.tem_pos_emb[:src_len, :] # [bs, src_len, emb_dim]
        src_mask = word_tokens.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        # print("skel_len: ", skel_len)
        # print("word_len: ", word_len)

        max_len = max(skel_len)
        tgt_pad_token = points_tokens.new(1, 3).fill_(self.pad_idx)
        
        min_num_masks = 1
        tgt_mask_inp = []
        tgt_mask_out = []
        tgt_vanilla_inp = []
        tgt_vanilla_out = []
        end_len = start_len = 0
        for i in range(bs):
            cur_len = skel_len[i].item()
            end_len += cur_len
            cur_point = points_tokens[start_len:end_len] # [cur_len, 3]
            
            cur_vanilla_inp = torch.ones_like(cur_point).to(cur_point.device) * self.mask_idx
            cur_vanilla_inp = torch.cat([cur_vanilla_inp, tgt_pad_token.repeat(max_len - cur_len, 1)], dim=0) # [max_len, 3]
            cur_vanilla_out = torch.cat([cur_point, tgt_pad_token.repeat(max_len - cur_len, 1)], dim=0) # [max_len, 3]
            
            tgt_vanilla_inp.append(cur_vanilla_inp.unsqueeze_(0))
            tgt_vanilla_out.append(cur_vanilla_out.unsqueeze_(0))

            cur_point = cur_point.view(-1)  # [cur_len * 3]
            sample_size = self.random.randint(min_num_masks, cur_len*3)
            ind = self.random.choice(cur_len*3, size=sample_size, replace=False)

            cur_inp = cur_point.clone()
            cur_inp[ind] = self.mask_idx
            cur_inp = cur_inp.view(cur_len, 3)
            cur_inp = torch.cat([cur_inp, tgt_pad_token.repeat(max_len - cur_len, 1)], dim=0) # [max_len, 3]
            tgt_mask_inp.append(cur_inp.unsqueeze_(0))

            cur_out = torch.ones_like(cur_point).to(cur_point.device) * self.pad_idx
            cur_out[ind] = cur_point[ind]
            cur_out = cur_out.view(cur_len, 3)
            cur_out = torch.cat([cur_out, tgt_pad_token.repeat(max_len - cur_len, 1)], dim=0) # [max_len, 3]
            tgt_mask_out.append(cur_out.unsqueeze_(0)) # [1, max_len, 3]
            start_len = end_len
        
        tgt_vanilla_inp = torch.cat(tgt_vanilla_inp, dim=0)
        tgt_vanilla_out = torch.cat(tgt_vanilla_out, dim=0)

        tgt_mask_inp = torch.cat(tgt_mask_inp, dim=0)
        tgt_mask_out = torch.cat(tgt_mask_out, dim=0)
        
        mask_loss, mask_ce_loss, mask_rec_loss, mask_logits, tgt_mask_no_pad  = self.teacher.get_loss_and_logits(src_feat, src_mask, tgt_mask_inp, tgt_mask_out, skel_len, mode, "mask", points_tokens, batch, self.vis, self.vis_token)
        vanilla_loss, vanilla_ce_loss, vanilla_rec_loss, vanilla_logits, tgt_vanilla_no_pad  = self.student.get_loss_and_logits(src_feat, src_mask, tgt_vanilla_inp, tgt_vanilla_out, skel_len, mode, "vanilla", points_tokens, batch, self.vis, self.vis_token)
        self.log('{}/mask_ce_loss'.format(mode), mask_ce_loss.detach(), prog_bar=True)  
        self.log('{}/mask_rec_loss'.format(mode), mask_rec_loss.detach(), prog_bar=True)  
        self.log('{}/vanilla_ce_loss'.format(mode), vanilla_ce_loss.detach(), prog_bar=True)  
        self.log('{}/vanilla_rec_loss'.format(mode), vanilla_rec_loss.detach(), prog_bar=True)  
        
        kl_loss = F.kl_div(F.log_softmax(vanilla_logits, dim=1),
						   F.softmax(mask_logits, dim=1),
						   reduction='none')
        kl_loss = kl_loss[tgt_mask_no_pad] 
        kl_loss = kl_loss.sum() / tgt_mask_no_pad.sum()

        self.log('{}/kl_loss'.format(mode), kl_loss.detach(), prog_bar=True)  
        loss = mask_loss + vanilla_loss + kl_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx, "val")


    @torch.no_grad()
    def generate(self, batch, batch_idx, total_seq_len=200, temperature = 1., default_batch_size = 1, ):
        self.vqvae.eval()
        word_tokens = batch["gloss_id"]
        word_len = batch["gloss_len"]
        bs, src_len = word_tokens.size()
        
        src_feat = self.src_embedding(word_tokens) + self.tem_pos_emb[:src_len, :] # [bs, src_len, emb_dim]
        src_mask = word_tokens.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)  # [bs, src_len]

        tgt_len = word_len * 8  # [bs]
        max_len = src_len * 8

        tgt_feat = self.tem_pos_emb[:max_len, :].unsqueeze_(0).repeat(bs, 1, 1)
        tgt_mask = self._get_mask(tgt_len, max_len, tgt_feat.device).unsqueeze_(1).unsqueeze_(2)

        pred_emb = self.transformer(src_feat, src_mask, tgt_feat, tgt_mask) # [bs, max_len, emd_dim]
        pred_emb = self.out_linear(pred_emb) # [bs, max_len+1, 3*emd_dim]

        pred_emb = einops.rearrange(pred_emb, "b t (n h) -> b t n h", n=3)
        logits_out = torch.matmul(pred_emb, self.tgt_embedding.weight.t()) # [bs, max_len, 3, n_codes+2]
        logits_out = logits_out[..., :-2] # [bs, max_len, 3, n_codes]

        probs = F.softmax(logits_out, dim=-1)
        max_probs, tgt_preds = probs.max(dim=-1)
        
        batch_preds = []
        for k in range(bs):
            predicts = tgt_preds[k, :tgt_len[k], :]
            predicts = F.embedding(predicts, self.vqvae.codebook.embeddings)  # [t, n, h]

            predicts = einops.rearrange(predicts, " t n h -> t h n")

            pred_pose, pred_rhand, pred_lhand = self.vqvae.decode(predicts)
            pred_points = torch.cat([pred_pose, pred_rhand, pred_lhand], dim=-1).detach().cpu().numpy() # [t, 150]
            
            batch_preds.append(pred_points)
            show_img = []
            for j in range(pred_points.shape[0]):
                frame_joints = pred_points[j]
                frame = np.ones((256, 256, 3), np.uint8) * 255
                frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
                # Draw the frame given 2D joints
                im = draw_frame_2D(frame, frame_joints_2d)
                show_img.append(im)
            show_img = np.concatenate(show_img, axis=1) # [h, w*16, c]
            cv2.imwrite("Data/predictions/nat/{}_{}.png".format(batch_idx, k), show_img)
        return tgt_preds, batch_preds 


    def vis_token(self, pred_tokens, mode, name):
        pred_tokens = " ".join([str(x) for x in pred_tokens.cpu().numpy().tolist()])
        self.logger.experiment.add_text("{}/{}_points".format(mode, name), pred_tokens, self.current_epoch)

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
        return points_tokens, points_embedding, skel_len, commitment_loss # [sum(skel_len), 3], # [sum(skel_len), emb_dim, 3]


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
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')

        parser.add_argument('--ctc_model', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--ctc_hparams_file', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--decoding_iterations', type=str, default=10, help='path to vqvae ckpt, or model name to download pretrained')

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

        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=256)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
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
        # casual_mask = self.casual_mask[:, :, :t, :t]
        # if tgt_mask is not None:
        #     pad_future_mask = casual_mask & tgt_mask
        # else:
        #     pad_future_mask = casual_mask
        
        dec_out = self.decoder(tgt_feat, tgt_mask, enc_out, src_mask)
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