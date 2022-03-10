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
from modules.transformer import TransformerEncoder, TransformerDecoder
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
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()
        
        max_source_positions, max_target_positions = 400, 400
        # encoder
        self.encoder = TransformerEncoder(len(text_dict), text_dict.pad(), max_target_positions, 
            hidden_size=640, ff_size=1280, num_heads=8, num_layers=6, dropout=0.1, emb_dropout=0.1)

        # dec_token_emb = nn.parameter(self.vqvae.codebook.embeddings)
        self.points_mask = self.vqvae.args.n_codes # 1024
        self.points_pad = self.vqvae.args.n_codes + 1 # 1025
        vocab_size = self.vqvae.args.n_codes + 2
        self.decoder = TransformerDecoder(vocab_size, self.points_pad, num_layers=6, 
            num_heads=8, hidden_size=640, ff_size=1280, dropout=0.1, emb_dropout=0.1)

        self.random = np.random.RandomState(seed)
        self.eps = 0.1

        # inference
        self.decoding_strategy = MaskPredict(decoding_iterations=5, token_num=self.token_num)

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

    @torch.no_grad()
    def _points2tokens(self, batch):
        pose = batch["pose"]
        bs, c, t, _ = pose.size()
        points_tokens, points_embedding, _ = self.vqvae.encode(batch) # [bs, t, self.token_num*4]
        pose_tokens = points_tokens[:, :, 0:5].contiguous().view(bs, -1)
        face_tokens = points_tokens[:, :, 5:10].contiguous().view(bs, -1)
        rhand_tokens = points_tokens[:, :, 10:15].contiguous().view(bs, -1)
        lhand_tokens = points_tokens[:, :, 15:20].contiguous().view(bs, -1)  # [bs, t//4*5]

        return pose_tokens, face_tokens, rhand_tokens, lhand_tokens, points_embedding

    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        self.vqvae.eval()
        # print("pose: ", batch["pose"].shape)

        pose_tokens, face_tokens, rhand_tokens, lhand_tokens, points_embedding = self._points2tokens(batch)
        # print("tgt_tokens: ", pose_tokens)

        points_len = batch["points_len"].long()

        word_tokens = batch["tokens"].long()
        bsz, _ = word_tokens.size()
        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat, predicted_lengths_lprobs = self.encoder(word_tokens=word_tokens, mask=word_mask)

        # length loss 
        length_target = batch["points_len"].long().unsqueeze(-1)
        assert max(length_target) < predicted_lengths_lprobs.size(-1)
        length_loss = -predicted_lengths_lprobs.gather(dim=-1, index=length_target)
        length_loss = length_loss.sum() / bsz * 0.05
        # pose loss
        # print("pose_len: ", points_len, length_target)
        # exit()

        pose_loss = self.separate_enc_dec("pose", pose_tokens, points_len, word_feat, word_mask, self.points_pad, self.points_mask)
        face_loss = self.separate_enc_dec("face", face_tokens, points_len, word_feat, word_mask, self.points_pad, self.points_mask)
        rhand_loss = self.separate_enc_dec("rhand", rhand_tokens, points_len, word_feat, word_mask, self.points_pad, self.points_mask)
        lhand_loss = self.separate_enc_dec("lhand", lhand_tokens, points_len, word_feat, word_mask, self.points_pad, self.points_mask)

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
            pose_recon, face_recon, rhand_recon, lhand_recon = self.vqvae.decode(points_embedding)
            pose_recon, face_recon = pose_recon[:, :, :vis_len, :], face_recon[:, :, :vis_len, :]
            rhand_recon, lhand_recon = rhand_recon[:, :, :vis_len, :], lhand_recon[:, :, :vis_len, :]
            self.vis("train", "rec_vis", pose_recon, face_recon, rhand_recon, lhand_recon)
        
        return loss

    def separate_enc_dec(self, tag_name, tgt_tokens, tgt_len, src_feat, src_mask, pad_idx, mask_idx):  
        min_num_masks = 1

        bs, T = tgt_tokens.size()
        # print("tgt_tokens: ", tgt_tokens.size())
        tgt_tokens = tgt_tokens.view(bs, T//5, 5)
        # print("tgt_tokens: ", tgt_tokens.shape)

        tgt_inp = tgt_tokens.clone() # [bs, t, 5]
        tgt_cp = tgt_tokens.clone()  # [bs, t, 5]
        tgt_out = torch.ones_like(tgt_tokens).to(tgt_tokens.device) * pad_idx

        # print("tgt_len: ", tgt_len)
        # for i in range(bs):
        #     length = tgt_len[i].cpu().item()
        #     if tag_name is not None: tgt_tokens[i, length:, :] = pad_idx
        #     sample_size = self.random.randint(min_num_masks, length)
        #     ind = self.random.choice(length, size=sample_size, replace=False)
        #     tgt_inp[i, ind, :] = mask_idx
        #     tgt_out[i, ind, :] = tgt_cp[i, ind, :]
        for i in range(bs):
            length = tgt_len[i].cpu().item()
            if tag_name is not None: tgt_tokens[i, length:, :] = pad_idx
            sample_size = self.random.randint(min_num_masks, length)
            ind = self.random.choice(length, size=sample_size, replace=False)
            tgt_inp[i, ind, :] = mask_idx
            tgt_out[i, ind, :] = tgt_cp[i, ind, :]
        
        # print("tgt_tokens: ", tgt_tokens.shape, tgt_tokens[:2,:10, :])
        # print("tgt_inp: ", tgt_inp.shape, tgt_inp[:2, :10, :])
        # print("tgt_out: ", tgt_out.shape, tgt_out[:2, :10, :])
        # exit()
        # print("tgt_len: ", tgt_len, tgt_inp.shape)
        size = max(tgt_len)
        tgt_mask = self._get_mask(tgt_len, size)
        # print("tgt_len: ", tgt_len)
        # print("tgt_mask: ", tgt_mask.shape)
        
        logits = self.decoder(trg_tokens=tgt_inp, encoder_output=src_feat, src_mask=src_mask, trg_mask=tgt_mask, 
                            mask_future=True, window_mask_future=False, window_size=self.token_num, tag_name=tag_name)
        # print("logits: ", logits.shape)
        
        # nll loss function
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1)) 
        target = tgt_out.view(-1, 1)
        non_pad_mask = target.ne(pad_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        reduce = True
        if reduce:
            tokens_nums = non_pad_mask.sum()
            assert tokens_nums != 0
            nll_loss = nll_loss.sum() / tokens_nums
            smooth_loss = smooth_loss.sum() / tokens_nums

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        
        return loss
        

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss


    def validation_step(self, batch, batch_idx):
        self.forward(batch, "val")
        if batch_idx < 1:
            self.inference_fast(batch)


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


    def inference_fast(self, batch):
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

        
        # predict_len = predict_len * self.token_num 
        print("predict_len and real length: ", predict_len, batch["points_len"].long())

        max_len = predict_len.max().item()

        init_mask = torch.arange(max_len).unsqueeze_(0).repeat(bsz, 1).to(predict_len.device)
        init_mask = (init_mask < (predict_len).unsqueeze(1)).bool()
        
        init_tokens = word_tokens.new(bsz, max_len).fill_(self.points_mask)
        init_tokens = (1 - init_mask.long()) * init_tokens + init_mask.long() * self.points_pad
        
        print("infer dec: ", init_tokens.shape, word_feat.shape, word_mask.shape, init_mask.shape)
        exit()
        pose_tokens = self.decoding_strategy.generate_separate(self, "pose", init_tokens, word_feat, word_mask, init_mask, self.points_pad, self.points_mask)
        face_tokens = self.decoding_strategy.generate_separate(self, "face", init_tokens, word_feat, word_mask, init_mask, self.points_pad, self.points_mask)
        rhand_tokens = self.decoding_strategy.generate_separate(self, "rhand", init_tokens, word_feat, word_mask, init_mask, self.points_pad, self.points_mask)
        lhand_tokens = self.decoding_strategy.generate_separate(self, "lhand", init_tokens, word_feat, word_mask, init_mask, self.points_pad, self.points_mask)
        
        for idx in range(bsz):
            pose_tokens_vis = pose_tokens[idx:idx+1, :predict_len[idx]].contiguous().view(1, -1, self.token_num)
            face_tokens_vis = face_tokens[idx:idx+1, :predict_len[idx]].contiguous().view(1, -1, self.token_num)
            rhand_tokens_vis = rhand_tokens[idx:idx+1, :predict_len[idx]].contiguous().view(1, -1, self.token_num)
            lhand_tokens_vis = lhand_tokens[idx:idx+1, :predict_len[idx]].contiguous().view(1, -1, self.token_num)

            prediction_vis = torch.cat([pose_tokens_vis, face_tokens_vis, rhand_tokens_vis, lhand_tokens_vis], dim=-1) # [bs, t//4, 20]
            prediction_vis = prediction_vis.view(1, -1)

            if not (prediction_vis >= self.points_mask).any():
                predictions_emb = self.vqvae.codebook.dictionary_lookup(prediction_vis)
                predictions_emb = predictions_emb.permute(0, 2, 1).contiguous()
                predictions_emb = predictions_emb.view(1, 256, -1, 20)
                
                pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(predictions_emb)
                self.vis("val", "pred_vis_{}".format(idx), pose_pred, face_pred, rhand_pred, lhand_pred)

        return pose_tokens, face_tokens, rhand_tokens, lhand_tokens, predict_len

    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand, log=False)
        # self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        # return [optimizer]
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


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
        self, vocab_size,  points_pad, num_layers, num_heads, hidden_size, ff_size, dropout, emb_dropout):
        super(TransformerDecoder, self).__init__()

        # self.max_target_positions = max_target_positions
        self._hidden_size = hidden_size
        self._output_size = vocab_size

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

        self.point_tok_embedding = WordEmbeddings(embedding_dim=hidden_size//5, vocab_size=vocab_size, 
            pad_idx=points_pad, num_heads=8, norm_type=None, activation_type=None, scale=False, scale_factor=None)

        # self.learn_pe = nn.Embedding(self.max_target_positions + points_pad + 1, 512, points_pad)
        # nn.init.normal_(self.learn_pe.weight, mean=0, std=0.02)
        # nn.init.constant_(self.learn_pe.weight[points_pad], 0)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size//5, self._output_size, bias=False)

        self.register_buffer("window_subsequen_mask", window_subsequent_mask(2200, 20))


    def forward(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, tag_name=None):
        """x: trg_embed
        """
        # assert trg_mask is not None, "trg_mask required for Transformer"
        
        bsz, tgt_len, _ = trg_tokens.size()
        x = self.point_tok_embedding(trg_tokens, trg_mask)
        x = x.view(bsz, tgt_len ,-1)
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
        x = x.view(bsz, tgt_len, 5, -1)
        # print("out x: ", x.shape)
        x = self.output_layer(x)
        # print("out x: ", x.shape)
        # exit()
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




    
