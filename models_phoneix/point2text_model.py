from email.policy import default
from turtle import forward
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from modules.st_gcn import ConvTemporalGraphical, Graph
import torch.distributed as dist
import argparse
from modules.utils import shift_dim
import numpy as np
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.attention import Transformer, FeedForward
from modules.nearby_attn import AttnBlock
from modules.vq_fn import Codebook
import einops
from modules.sp_layer import SPL
from util.plot_videos import draw_frame_2D
from ctc_decoder import beam_search, best_path
from collections import defaultdict
from util.wer import get_wer_delsubins



class Point2textModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

        self.points_emb = nn.Linear(150, 256)

        self.vit = Encoder(dim=256, depth=4, heads=8, mlp_dim=1024, dropout = 0.)

        self.ctc_out = nn.Linear(256, len(text_dict))

        self.ctcLoss = nn.CTCLoss(text_dict.blank(), reduction="mean", zero_infinity=True)

        self.chars = [chr(x) for x in range(20000, 20000 + len(text_dict))]
        self.char_to_idx = defaultdict(int)
        for idx, c in enumerate(self.chars):
            self.char_to_idx[c] = idx 


    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        word_tokens = batch["gloss_id"]
        word_len = batch["gloss_len"]

        points = batch["skel_3d"]
        skel_len = batch["skel_len"]
        
        max_len = max(skel_len)

        points = self.points_emb(points)
        mask = self._get_mask(skel_len, max_len, points.device).unsqueeze_(1).unsqueeze_(2)

        points = self.vit(points, mask)

        logits = self.ctc_out(points)  # [bs, t, vocab_size]

        lprobs = logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        
        ctc_loss = self.ctcLoss(lprobs.cpu(), word_tokens.cpu(), skel_len.cpu(), word_len.cpu()).to(lprobs.device)
        self.log('{}/ctc_loss'.format(mode), ctc_loss.detach(), prog_bar=True)

        
        return ctc_loss, logits

    def training_step(self, batch):
        loss, _ = self.forward(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, logits = self.forward(batch, "val") # [bs, t, vocab_size]
        skel_len = batch["skel_len"]
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        bs = skel_len.size(0)
        probs = F.softmax(logits, dim=-1)
        
        correct = 0
        count = 0
        val_wer = np.zeros([4])
        for i in range(bs):
            cur_probs = probs[i, :skel_len[i], :].cpu().numpy() # [t, vocab_size]
            pred_seq = beam_search(mat=cur_probs, chars=self.chars, beam_width=5, blank_idx=self.text_dict.blank())

            hyp = [self.char_to_idx[c] for c in pred_seq]
            ref = gloss_id[i][:gloss_len[i]].cpu().numpy().tolist()

            correct += int(ref == hyp)
            wer = get_wer_delsubins(ref, hyp)
            val_wer += np.array(wer)
            count += 1
        return dict(wer=val_wer, correct=correct, count=count)

    def validation_epoch_end(self, outputs) -> None:
        val_err, val_correct, val_count = np.zeros([4]), 0, 0
        for out in outputs:
            val_err += out["wer"]
            val_correct += out["correct"]
            val_count += out["count"]

        self.log('{}/acc'.format("val"), val_correct / val_count, prog_bar=True)
        self.log('{}/wer'.format("val"), val_err[0] / val_count, prog_bar=True)
        self.log('{}/sub'.format("val"), val_err[1] / val_count, prog_bar=True)
        self.log('{}/ins'.format("val"), val_err[2] / val_count, prog_bar=True)
        self.log('{}/del'.format("val"), val_err[3] / val_count, prog_bar=True)
        
    
    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96, last_epoch=-1)
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser





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