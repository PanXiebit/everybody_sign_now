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




class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict, seed=888):
        super().__init__()
        self.args = args
        self.text_dict = text_dict
        
        self.token_num = 5
        self.max_target_positions = args.max_frames_num + 1
        self.max_source_positions = 500

        # Load VQ-VAE and set all parameters to no grad
        from .pose_vqvae_vit_model_mcodebooks import PoseVitVQVAE
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  PoseVitVQVAE.load_from_checkpoint(args.pose_vqvae, hparams_file=args.hparams_file)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        for codebook in self.vqvae.codebooks:
            codebook._need_init = False
        self.vqvae.eval()
        
        max_source_positions, max_target_positions = 400, 400

        # encoder
        self.encoder = TransformerEncoder(len(text_dict), text_dict.pad(), max_target_positions, 
            hidden_size=1280, ff_size=2560, num_heads=5, num_layers=4, dropout=0.1, emb_dropout=0.1)

        embed_dim = self.vqvae.args.embedding_dim
        self.bos = nn.Parameter(torch.zeros(1, 1, 1, embed_dim), requires_grad=True)
        self.eos = nn.Parameter(torch.zeros(1, 1, 1, embed_dim), requires_grad=True)
        self.pad = nn.Parameter(torch.zeros(1, 1, 1, embed_dim), requires_grad=True)

        self.decoder = TransformerDecoder(num_layers=4, num_heads=5, hidden_size=1280, ff_size=2560, dropout=0.1, emb_dropout=0.1)

        self.random = np.random.RandomState(seed)
        self.eps = 0.1

        # inference
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

    @torch.no_grad()
    def _points2tokens(self, batch):
        pose = batch["pose"]
        bs, c, t, _ = pose.size()
        points_tokens, points_embedding, _ = self.vqvae.encode(batch) # [bs, 256, t, self.token_num*4]
        return points_embedding

    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        self.vqvae.eval()
        points_len = batch["points_len"].long()
        # print("points_len: ", points_len)
        points_embedding_vis = self._points2tokens(batch) # [bs, 256, t, 20]

        points_embedding = points_embedding_vis.clone().permute(0, 2, 3, 1).contiguous() # [bs, t, 20, 256]

        word_tokens = batch["tokens"].long()
        bsz, _ = word_tokens.size()
        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat, predicted_lengths_lprobs = self.encoder(word_tokens=word_tokens, mask=word_mask)

        # length loss 
        length_target = batch["points_len"].long().unsqueeze(-1)
        assert max(length_target) < predicted_lengths_lprobs.size(-1)
        length_loss = -predicted_lengths_lprobs.gather(dim=-1, index=length_target)
        length_loss = length_loss.sum() / bsz * 0.05
        
        
        points_inp = torch.cat([self.bos.repeat(bsz, 1, 20, 1), points_embedding], dim=1) # [bs, t+1, 20, 256]
        points_out = torch.cat([points_embedding, self.pad.repeat(bsz, 1, 20, 1)], dim=1) # [bs, t+1, 20, 256]
        for b in range(bsz):
            length = points_len[b]
            points_inp[b, length+1:, :, :] = self.pad.repeat(1, 1, 20, 1)
            points_out[b, length+1:, :, :] = self.pad.repeat(1, 1, 20, 1)
            points_out[b, length:length+1, :, :] = self.eos.repeat(1, 1, 20, 1)
        points_len = points_len + 1

        t = points_inp.size(1)
        pose_emb_inp = points_inp[:, :, 0:5, :].view(bsz, t, -1)
        face_emb_inp = points_inp[:, :, 5:10, :].view(bsz, t, -1)
        rhand_emb_inp = points_inp[:, :, 10:15, :].view(bsz, t, -1)
        lhand_emb_inp = points_inp[:, :, 15:20, :].view(bsz, t, -1)


        pose_emb_out = points_out[:, :, 0:5, :].view(bsz, t, -1)
        face_emb_out = points_out[:, :, 5:10, :].view(bsz, t, -1)
        rhand_emb_out = points_out[:, :, 10:15, :].view(bsz, t, -1)
        lhand_emb_out = points_out[:, :, 15:20, :].view(bsz, t, -1)

        pose_emb_pred = self.separate_enc_dec("pose", pose_emb_inp, points_len, word_feat, word_mask, pose_emb_out)
        face_emb_pred = self.separate_enc_dec("face", face_emb_inp, points_len, word_feat, word_mask, face_emb_out)
        rhand_emb_pred = self.separate_enc_dec("rhand", rhand_emb_inp, points_len, word_feat, word_mask, rhand_emb_out)
        lhand_emb_pred = self.separate_enc_dec("lhand", lhand_emb_inp, points_len, word_feat, word_mask, lhand_emb_out)

        points_emb_pred = torch.cat([pose_emb_pred, face_emb_pred, rhand_emb_pred, lhand_emb_pred], dim=-1)
        pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(points_emb_pred)
        
        pose = self.vqvae.selects(batch["pose"], "pose")
        face = self.vqvae.selects(batch["face"], "face")
        rhand = self.vqvae.selects(batch["rhand"], "rhand")
        lhand = self.vqvae.selects(batch["lhand"], "lhand")
        
        pose_no_mask = self.vqvae.selects(batch["pose_no_mask"], "pose")
        face_no_mask = self.vqvae.selects(batch["face_no_mask"], "face")
        rhand_no_mask = self.vqvae.selects(batch["rhand_no_mask"], "rhand")
        lhand_no_mask = self.vqvae.selects(batch["lhand_no_mask"], "lhand")

        # print("pose: ", pose.shape, face.shape, rhand.shape, lhand.shape)
        # print("pose_no_mask: ", pose_no_mask.shape, face_no_mask.shape, rhand_no_mask.shape, lhand_no_mask.shape)
        # exit()

        pose_loss = (torch.abs(pose - pose_pred) * pose_no_mask).sum() / (pose_no_mask.sum() + 1e-7)
        # print("pose_rec_loss: ", pose_rec_loss.shape, pose_no_mask.shape)
        face_loss = (torch.abs(face - face_pred) * face_no_mask).sum() / (face_no_mask.sum() + 1e-7)
        rhand_loss = (torch.abs(rhand - rhand_pred) * rhand_no_mask).sum() / (rhand_no_mask.sum()+ 1e-7)
        lhand_loss = (torch.abs(lhand - lhand_pred) * lhand_no_mask).sum() / (lhand_no_mask.sum() + 1e-7)


        loss = length_loss + pose_loss + face_loss + rhand_loss + lhand_loss
        
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        self.log('{}/length_loss'.format(mode), length_loss.detach(), prog_bar=True)
        self.log('{}/pose_loss'.format(mode), pose_loss.detach(), prog_bar=True)
        self.log('{}/face_loss'.format(mode), face_loss.detach(), prog_bar=True)
        self.log('{}/rhand_loss'.format(mode), rhand_loss.detach(), prog_bar=True)
        self.log('{}/lhand_loss'.format(mode), lhand_loss.detach(), prog_bar=True)
        self.log('{}/learning_rate'.format(mode), self.get_lr(), prog_bar=True)

        if self.global_step % 500 == 0:
            # original 
            vis_len = batch["points_len"].long()[0]
            pose = self.vqvae.selects(batch["pose"], "pose")[:, :, :vis_len, :]
            face = self.vqvae.selects(batch["face"], "face")[:, :, :vis_len, :]
            rhand = self.vqvae.selects(batch["rhand"], "rhand")[:, :, :vis_len, :]
            lhand = self.vqvae.selects(batch["lhand"], "lhand")[:, :, :vis_len, :]
            self.vis("train", "ori_vis", pose, face, rhand, lhand)
            # reconstrction
            pose_recon, face_recon, rhand_recon, lhand_recon = self.vqvae.decode(points_embedding_vis)

            pose_recon, face_recon = pose_recon[:, :, :vis_len, :], face_recon[:, :, :vis_len, :]
            rhand_recon, lhand_recon = rhand_recon[:, :, :vis_len, :], lhand_recon[:, :, :vis_len, :]
            self.vis("train", "rec_vis", pose_recon, face_recon, rhand_recon, lhand_recon)
        
        return loss

    def separate_enc_dec(self, tag_name, tgt_emb_inp, tgt_len, src_feat, src_mask, tgt_emb_out): 
        bsz, _, _ = tgt_emb_inp.size() 
        
        size = max(tgt_len)
        tgt_mask = self._get_mask(tgt_len, size)
        pred_emb = self.decoder(trg_tokens=tgt_emb_inp, encoder_output=src_feat, src_mask=src_mask, trg_mask=tgt_mask, 
                                mask_future=True, window_mask_future=False, window_size=self.token_num, tag_name=tag_name)
        pred_emb = pred_emb[:, :-1, :].view(bsz, -1, 5, 256).permute(0, 3, 1, 2).contiguous()
        # print("pred_emb: ", pred_emb.shape)
        # exit()
        # embed_size = pred_emb.size(-1)
        # # print("tgt_emb_out, pred_emb: ", tgt_emb_out.shape, pred_emb.shape)

        # dist = F.cosine_similarity(tgt_emb_out.view(-1, embed_size), pred_emb.view(-1, embed_size))
        # loss = (1 - dist) / 2.
        # loss = loss[tgt_mask.view(-1)].sum() / tgt_mask.view(-1).sum()
        return pred_emb
        

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


    def greedy_dec(self, word_feat, word_mask, max_len, tag_name):
        bsz = word_feat.size(0)
        ys_emb = self.bos.repeat(bsz, 1, 5, 1).view(bsz, 1, -1) # [bs, 1, 1280]
        for i in range(max_len -1):
            pred_emb = self.decoder(ys_emb, encoder_output=word_feat, src_mask=word_mask, trg_mask=None, 
                                    mask_future=True, window_mask_future=False, window_size=self.token_num, tag_name=tag_name)
            ys_emb = torch.cat([ys_emb, pred_emb[:, -1:, :]], dim=1)
        return ys_emb.view(bsz, max_len, 5, -1)


    def inference_greedy(self, batch):
        self.vqvae.eval()
        word_tokens = batch["tokens"].long()
        bsz = word_tokens.size(0)
        for id in range(bsz):
            vis_len = batch["points_len"].long()[id]
            pose = self.vqvae.selects(batch["pose"], "pose")[id:id+1, :, :vis_len, :]
            face = self.vqvae.selects(batch["face"], "face")[id:id+1, :, :vis_len, :]
            rhand = self.vqvae.selects(batch["rhand"], "rhand")[id:id+1, :, :vis_len, :]
            lhand = self.vqvae.selects(batch["lhand"], "lhand")[id:id+1, :, :vis_len, :]
            self.vis("val", "ori_vis_{}".format(id), pose, face, rhand, lhand)

        word_mask = word_tokens.ne(self.text_dict.pad())
        # print("word_tokens: ", word_tokens[:, :10])
        word_feat, predicted_lengths_probs = self.encoder(word_tokens=word_tokens, mask=word_mask)
        predict_len = torch.argmax(predicted_lengths_probs, dim=-1) # [bs]
        predict_len[predict_len < 2] = 2

        # print("predict_len and real length: ", predict_len, batch["points_len"].long())

        max_len = predict_len.max().item()
        pose_pred_emb = self.greedy_dec(word_feat, word_mask, max_len, "pose")
        face_pred_emb = self.greedy_dec(word_feat, word_mask, max_len, "face")
        rhand_pred_emb = self.greedy_dec(word_feat, word_mask, max_len, "rhand")
        lhand_pred_emb = self.greedy_dec(word_feat, word_mask, max_len, "lhand")
        
        # print("pose_pred_emb, face_pred_emb, rhand_pred_emb, lhand_pred_emb: ", pose_pred_emb.shape, face_pred_emb.shape, rhand_pred_emb.shape, lhand_pred_emb.shape)
        for idx in range(bsz):
            prediction_vis = torch.cat([pose_pred_emb, face_pred_emb, rhand_pred_emb, lhand_pred_emb], dim=-2) # [bs, t, 20, 256]
            predictions_vis = prediction_vis.permute(0, 3, 1, 2).contiguous()
            pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(predictions_vis)
            self.vis("val", "pred_vis_{}".format(idx), pose_pred, face_pred, rhand_pred, lhand_pred)


    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand, log=False)
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
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
        self, num_layers, num_heads, hidden_size, ff_size, dropout, emb_dropout):
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
        self.register_buffer("window_subsequen_mask", window_subsequent_mask(2200, 20))


    def forward(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, tag_name=None):
        """x: trg_embed
        """
        # assert trg_mask is not None, "trg_mask required for Transformer"
        x = trg_tokens
        bsz, tgt_len, _ = x.size()
        x = x + self.abs_pe(x)

        if tag_name is not None:
            if tag_name == "pose":
                x = x + self.tag_emb[0].repeat(bsz, tgt_len, 1)
            elif tag_name == "face":
                x = x + self.tag_emb[1].repeat(bsz, tgt_len, 1)
            elif tag_name == "rhand":
                x = x + self.tag_emb[2].repeat(bsz, tgt_len, 1)
            elif tag_name == "lhand":
                x = x + self.tag_emb[3].repeat(bsz, tgt_len, 1)
            else:
                raise ValueError("{} is wrong!".format(tag_name))

        x = self.emb_dropout(x)

        if mask_future:
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & subsequent_mask(x.size(1)).bool().to(x.device)
            else:
                trg_mask = subsequent_mask(x.size(1)).bool().to(x.device)
        
        if window_mask_future:
            assert window_size is not None
            size = x.size(1)
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & self.window_subsequen_mask[:, :size, :size].to(x.device)
            else:
                trg_mask = self.window_subsequen_mask[:, :size, :size].to(x.device)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        return x

    def forward_fast(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, 
                     past_self=None):
        """x: trg_embed
        """
        # inference only
        assert not self.training

        # assert trg_mask is not None, "trg_mask required for Transformer"
        x = self.point_tok_embedding(trg_tokens, trg_mask)

        if past_self is not None:
            past_length = past_self.size(-2)
            assert past_length is not None

            # TODO:
            x = x + self.abs_pe(x, past_length)
        else:
            x = x + self.abs_pe(x)  # add position encoding to word embedding

        x = self.emb_dropout(x)

        presents_self = []  # accumulate over layers
        for i, layer in enumerate(self.layers):
            x, present_self = layer.forward_fast(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask, 
                layer_past_self=past_self[i, ...] if past_self is not None else None,
                return_present=True)
            
            presents_self.append(present_self)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, torch.stack(presents_self)







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




    
