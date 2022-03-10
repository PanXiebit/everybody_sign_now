import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random

import argparse
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from torchvision.utils import save_image
from modules.mask_predict import MaskPredict
from modules.transformer import TransformerEncoder
from modules.transformer.utils import BertLayerNorm, gelu, GeLU
from modules.transformer.multihead_attention import MultiHeadedAttention
from modules.transformer.position_encoding import PositionalEncoding
from modules.transformer.encoder import PositionwiseFeedForward
from modules.transformer.word_embedding import WordEmbeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data.data_prep.renderopenpose import *
import torchvision
import cv2



class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict, seed=888):
        super().__init__()
        self.args = args
        self.text_dict = text_dict
        
        self.max_target_positions = args.max_frames_num + 1
        self.max_source_positions = 500

        self.tokens = {}
        self.tokens["pose"] = list(range(8)) # 8
        self.tokens["rhand"] = list(range(21)) # 21
        self.tokens["lhand"] = list(range(21)) # 21

        emb_dim = 512
        self.linear_emb = nn.Linear(100, emb_dim)

        max_source_positions, max_target_positions = 400, 400

        # encoder
        self.encoder = TransformerEncoder(len(text_dict), text_dict.pad(), max_target_positions, 
            hidden_size=emb_dim, ff_size=emb_dim*4, num_heads=8, num_layers=4, dropout=0.1, emb_dropout=0.1)

        self.bos = nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True)
        self.pad = nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True)

        self.decoder = TransformerDecoder(output_size=100, num_layers=4, num_heads=8, hidden_size=emb_dim, ff_size=emb_dim*4, dropout=0.1, emb_dropout=0.1)

        self.random = np.random.RandomState(seed)
        self.eps = 0.1

        # inference
        self.apply(self.init_bert_weights)
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


    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        
        # encoder
        word_tokens = batch["tokens"].long()
        bsz, _ = word_tokens.size()
        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat, predicted_lengths_lprobs = self.encoder(word_tokens=word_tokens, mask=word_mask)

        # length loss 
        length_target = batch["points_len"].long().unsqueeze(-1)
        assert max(length_target) < predicted_lengths_lprobs.size(-1)
        length_loss = -predicted_lengths_lprobs.gather(dim=-1, index=length_target)
        length_loss = length_loss.sum() / bsz * 0.05


        points_len = batch["points_len"].long()

        bs, _, t, _ = batch["pose"].size()
        pose_ori = batch["pose"][..., self.tokens["pose"]]                 # [bs, 2, t, v]
        pose = pose_ori.permute(0, 2, 1, 3).contiguous().view(bs, t, -1)   # [bs, t, 16]
        rhand_ori = batch["rhand"][..., self.tokens["rhand"]]              # [bs, 2, t, v]
        rhand = rhand_ori.permute(0, 2, 1, 3).contiguous().view(bs, t, -1) # [bs, t, 42]
        lhand_ori = batch["lhand"][..., self.tokens["lhand"]]              # [bs, 2, t, v]
        lhand = lhand_ori.permute(0, 2, 1, 3).contiguous().view(bs, t, -1) # [bs, t, 42]
        points = torch.cat([pose, rhand, lhand], dim=-1) # [bs, t, 100]

        
        points_out = points.clone() # [bs, t, 100] 

        points_inp = points.clone()[:, :-1, :] # [bs, t-1, 100]
        point_inp_emb = self.linear_emb(points_inp) # [bs, t-1, 512]
        point_inp_emb = torch.cat([self.bos.repeat(bs, 1, 1), point_inp_emb], dim=1) # [bs, t, 100]

        tgt_mask = self._get_mask(points_len, t)
        pred_points, _ = self.decoder(point_inp_emb, word_feat, word_mask, tgt_mask, mask_future=True) # [bs, t, 100]
        reg_loss = F.mse_loss(pred_points, points_out, reduction="none")
        reg_loss = reg_loss.sum(-1).view(-1)[tgt_mask.view(-1)]
        reg_loss = reg_loss.sum() / tgt_mask.sum()
        

        loss = reg_loss + length_loss * 0.01
        
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        self.log('{}/length_loss'.format(mode), length_loss.detach(), prog_bar=True)
        self.log('{}/reg_loss'.format(mode), reg_loss.detach(), prog_bar=True)
        self.log('{}/learning_rate'.format(mode), self.get_lr(), prog_bar=True)

        

        if self.global_step % 500 == 0:
            # original
            pose_pred = pred_points[:, :, 0:8*2].view(bs, t, 2, -1).permute(0, 2, 1, 3).contiguous()
            rhand_pred = pred_points[:, :, 8*2:(8+21)*2].view(bs, t, 2, -1).permute(0, 2, 1, 3).contiguous()
            lhand_pred = pred_points[:, :, (8+21)*2:].view(bs, t, 2, -1).permute(0, 2, 1, 3).contiguous()

            # print("pose_ori, rhand_ori, lhand_ori: ", pose_ori.shape, rhand_ori.shape, lhand_ori.shape)
            # print("pose_pred, rhand_pred, lhand_pred: ", pose_pred.shape, rhand_pred.shape, lhand_pred.shape)
            
            # exit()

            for b in range(4):
                length = points_len[b]
                self.visualization("train", "ori_vis", pose_ori[b, :, :length, :], rhand_ori[b, :, :length, :], lhand_ori[b, :, :length, :]) # c, t, v
                self.visualization("train", "pred_vis", pose_pred[b, :, :length, :], rhand_pred[b, :, :length, :], lhand_pred[b, :, :length, :])
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.forward(batch, "val")
        if batch_idx < 1:
            self.inference_greedy(batch)

    def get_lr(self):
        for param_group in self.trainer.optimizers[0].param_groups:
            return param_group['lr']

    def make_positions(self, tensor, padding_idx):
        mask = tensor.ne(padding_idx).long()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask)

    def make_point_positions(self, tensor, padding_idx):
        mask = tensor.ne(padding_idx).long()
        positions =  (torch.cumsum(mask, dim=1).type_as(mask) * mask)
        return ((positions - 1) // self.token_num + 1).type_as(mask) * mask


    def greedy_dec(self, word_feat, word_mask, max_len):
        bsz = word_feat.size(0)
        ys_emb = self.bos.repeat(bsz, 1, 1) # [bs, 1, 512]
        for i in range(max_len):
            _, pred_emb = self.decoder(ys_emb, encoder_output=word_feat, src_mask=word_mask, trg_mask=None, mask_future=True) # [bs, t, 512]
            ys_emb = torch.cat([ys_emb, pred_emb[:, -1:, :]], dim=1)
        pred_emb = self.decoder.output_layer(ys_emb[:, 1:, :])
        return pred_emb


    def inference_greedy(self, batch):
        pose_ori = batch["pose"][..., self.tokens["pose"]]                 # [bs, 2, t, v]
        rhand_ori = batch["rhand"][..., self.tokens["rhand"]]              # [bs, 2, t, v]
        lhand_ori = batch["lhand"][..., self.tokens["lhand"]]              # [bs, 2, t, v]

        word_tokens = batch["tokens"].long()
        bs, _ = word_tokens.size()
        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat, predicted_lengths_lprobs = self.encoder(word_tokens=word_tokens, mask=word_mask)

        predict_len = torch.argmax(predicted_lengths_lprobs, dim=-1) # [bs]
        predict_len[predict_len < 2] = 2

        # print("predict_len and real length: ", predict_len, batch["points_len"].long())

        max_len = predict_len.max().item()
        pred_points = self.greedy_dec(word_feat, word_mask, max_len)
        
        pose_pred = pred_points[:, :, 0:8*2].view(bs, max_len, 2, -1).permute(0, 2, 1, 3).contiguous()
        rhand_pred = pred_points[:, :, 8*2:(8+21)*2].view(bs, max_len, 2, -1).permute(0, 2, 1, 3).contiguous()
        lhand_pred = pred_points[:, :, (8+21)*2:].view(bs, max_len, 2, -1).permute(0, 2, 1, 3).contiguous()

        points_len = batch["points_len"].long()

        for b in range(4):
            ori_length = points_len[b]
            pred_length = predict_len[b]
            self.visualization("val", "ori_vis", pose_ori[b, :, :ori_length, :], rhand_ori[b, :, :ori_length, :], lhand_ori[b, :, :ori_length, :])      # c, t, v
            self.visualization("val", "pred_vis", pose_pred[b, :, :pred_length, :], rhand_pred[b, :, :pred_length, :], lhand_pred[b, :, :pred_length, :])  # c, t, v



    def visualization(self, mode, name, pose, rhand, lhand, log=True):
        # visualize
        ori_vis = []

        c, t, v = pose.size()
        pose = pose.permute(1, 2, 0).contiguous()  # [t, v, c]
        rhand = rhand.permute(1, 2, 0).contiguous()
        lhand = lhand.permute(1, 2, 0).contiguous()

        for i in range(pose.size(0)):   
            pose_anchor = (640, 360)
            pose_list = self._tensor2numpy(pose[i], pose_anchor, "pose", 25) # [3V]

            face_anchor = (pose_list[0*3], pose_list[0*3 + 1])
            rhand_anchor = (pose_list[4*3], pose_list[4*3 + 1])
            lhand_anchor = (pose_list[7*3], pose_list[7*3 + 1])
            # rhand_anchor = pose_anchor
            # lhand_anchor = pose_anchor

            rhand_list = self._tensor2numpy(rhand[i], rhand_anchor, "rhand", 21)# , rhand_anchor[0] * 640, rhand_anchor[1] * 360)
            lhand_list = self._tensor2numpy(lhand[i], lhand_anchor, "lhand", 21) # , lhand_anchor[0] * 640, lhand_anchor[1] * 360)

            canvas = self._render(pose_list, rhand_list, lhand_list)
            canvas = torch.FloatTensor(canvas) # [h, w, c]
            canvas = canvas.permute(2, 0, 1).contiguous().unsqueeze(0)
            
            ori_vis.append(canvas) # [1, c, h, w]
        ori_vis = torch.cat(ori_vis, dim=0)
        ori_vis = torchvision.utils.make_grid(ori_vis)
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)
        return mode, name, ori_vis
    

    def _tensor2numpy(self, points, anchor, part_name, keypoint_num):
        """[v, c]]
        """
        points = points.detach().cpu().numpy()
        v, c = points.shape
        # [[17, 15, 0, 16, 18], [0, 1, 8, 9, 12], [4, 3, 2, 1, 5], [2, 1, 5, 6, 7]]
        pose_tokens = []
        for ids in self.tokens[part_name]:
            pose_tokens.append(ids)

        pose_vis = np.zeros((keypoint_num, 3), dtype=np.float32)
        for i in range(len(pose_tokens)):
            pose_vis[pose_tokens[i], 0] = points[i][0] * 1280 + anchor[0]
            pose_vis[pose_tokens[i], 1] = points[i][1] * 720 + anchor[1]
            pose_vis[pose_tokens[i], -1] = 1.
        
        pose_vis = pose_vis.reshape((-1, ))
        # print(pose_vis.shape, pose_vis)
        return pose_vis.tolist()


    def _render(self, posepts, r_handpts, l_handpts):
        myshape = (720, 1280, 3)
        numkeypoints = 70
        canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
        canvas = renderhand(r_handpts, canvas)
        canvas = renderhand(l_handpts, canvas) # [720, 720, 3]
        canvas = canvas[:, 280:1000, :]
        canvas = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC) # [256, 256, 3]
        return canvas # [256, 256, 3]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer]
        # return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


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


def window_subsequent_mask(size, window_size):
    pos = torch.arange(0, size).unsqueeze(0).repeat(size, 1)
    right = torch.arange(window_size-1, size, window_size).unsqueeze(1).repeat(1, window_size).view(size, 1)
    mask = (pos <= right)
    return mask.unsqueeze(0)

def subsequent_mask(size: int) -> Tensor:
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = BertLayerNorm(size, eps=1e-6)
        self.dec_layer_norm = BertLayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor = None, memory: Tensor = None, src_mask: Tensor = None, trg_mask: Tensor = None):        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)
        o = self.feed_forward(self.dropout(h2) + h1)

        return o

    def forward_fast(
        self, x: Tensor = None, memory: Tensor = None, src_mask: Tensor = None, trg_mask: Tensor = None, 
        layer_past_self=None, return_present=True):

        if return_present: assert not self.training

        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1, present_self = self.trg_trg_att.forward_fast(x_norm, x_norm, x_norm, mask=trg_mask, layer_past=layer_past_self)
        
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2, _ = self.src_trg_att.forward_fast(memory, memory, h1_norm, mask=src_mask, layer_past=None)
        o = self.feed_forward(self.dropout(h2) + h1)

        return o, present_self


class TransformerDecoder(nn.Module):
    def __init__(
        self, output_size, num_layers, num_heads, hidden_size, ff_size, dropout, emb_dropout):
        super(TransformerDecoder, self).__init__()

        # self.max_target_positions = max_target_positions
        self._hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.tag_emb = nn.Parameter(torch.randn(4, 1, hidden_size), requires_grad=True)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.register_buffer("window_subsequen_mask", window_subsequent_mask(2200, 20))


    def forward(self, trg_inp, encoder_output, src_mask, trg_mask, mask_future=True):
        """x: trg_embed
        """
        # assert trg_mask is not None, "trg_mask required for Transformer"
        bsz, tgt_len, _ = trg_inp.size()
        x = trg_inp + self.abs_pe(trg_inp)

        x = self.emb_dropout(x)

        if mask_future:
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & subsequent_mask(x.size(1)).bool().to(x.device)
            else:
                trg_mask = subsequent_mask(x.size(1)).bool().to(x.device)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        preds = self.output_layer(x)
        return preds, x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = BertLayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            GeLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = BertLayerNorm(size, eps=1e-6)
           
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x, mask):
        
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, max_target_positions, hidden_size, ff_size, num_heads, num_layers, dropout, emb_dropout):
        super(TransformerEncoder, self).__init__()
        # self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.padding_idx=pad_idx

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for num in range(num_layers)
            ]
        )
        self.word_embedding = WordEmbeddings(embedding_dim=hidden_size, vocab_size=vocab_size, 
            pad_idx=pad_idx, num_heads=8, norm_type="batch", activation_type="softsign", scale=True, scale_factor=None)

        self.layer_norm = BertLayerNorm(hidden_size, eps=1e-6)
        # self.learn_pe = nn.Embedding(self.max_source_positions + self.padding_idx + 1, 512, self.padding_idx)
        # nn.init.normal_(self.learn_pe.weight, mean=0, std=0.02)
        # nn.init.constant_(self.learn_pe.weight[self.padding_idx], 0)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # learn prediction
        self.embed_lengths = nn.Embedding(self.max_target_positions + 1, hidden_size)
        nn.init.normal_(self.embed_lengths.weight, mean=0, std=0.02)


    def forward(self, word_tokens, mask):
        """
        """
        if word_tokens.ndim == 2:
            x = self.word_embedding(word_tokens, mask)
        elif word_tokens.ndim == 3:
            x = word_tokens
            # print("the encoder input is embeded!")
        else:
            raise ValueError("word_token dim is not 2 or 3!")
            
        x = x + self.abs_pe(word_tokens) 

        # x = x + self.learn_pe(word_tokens)  # add position encoding to word embeddings
        x = self.emb_dropout(x)  # [bs, length, embed_size]
        len_tokens = self.embed_lengths(word_tokens.new(word_tokens.size(0), 1).long().fill_(0))
        x = torch.cat([len_tokens, x], dim=1)
        mask = torch.cat([mask.new(word_tokens.size(0), 1).fill_(1), mask], dim=1)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        x = x[:, 1:, :]
        mask = mask[:, 1:]

        predicted_lengths_logits = torch.matmul(x[:, 0, :], self.embed_lengths.weight.transpose(0, 1)).float()
        predicted_lengths_logits[:, 0] += float('-inf')   # Cannot predict the len_token
        predicted_lengths_lprobs = F.log_softmax(predicted_lengths_logits, dim=-1)
        return x, predicted_lengths_lprobs




    
