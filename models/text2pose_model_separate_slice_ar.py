import os
import itertools
import numpy as np
from pyrsistent import b
from tqdm import tqdm
import argparse
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torchvision
from torchvision.utils import save_image
from modules.mask_predict import MaskPredict
from modules.transformer import TransformerEncoder
import random
from modules.transformer.utils import BertLayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.transformer.multihead_attention import MultiHeadedAttention3D
from modules.transformer.position_encoding import PositionalEncoding
from modules.transformer.encoder import PositionwiseFeedForward
from modules.transformer.word_embedding import WordEmbeddings


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
        # self.vqvae.codebook._need_init = False
        self.vqvae.eval()
        
        max_source_positions, max_target_positions = 400, 400
        # encoder
        self.encoder = TransformerEncoder(len(text_dict), text_dict.pad(), max_target_positions, 
            hidden_size=256, ff_size=1024, num_heads=4, num_layers=3, dropout=0.1, emb_dropout=0.1)

        vocab_size = self.vqvae.args.n_codes * 20 + 3 # 5120
        self.points_bos = int(vocab_size - 3) # 5119
        self.points_eos = int(vocab_size - 2) # 5120
        self.points_pad = int(vocab_size - 1) # 5120
        self.decoder = TransformerDecoder(vocab_size, self.points_pad, num_layers=3, 
            num_heads=4, hidden_size=256, ff_size=1024, dropout=0.1, emb_dropout=0.1)

        self.random = np.random.RandomState(seed)
        self.eps = 0.1

        # inference
        # self.decoding_strategy = MaskPredict(decoding_iterations=10, token_num=self.token_num)

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
        with torch.no_grad():
            pose = batch["pose"]
            bs, c, t, _ = pose.size()
            points_tokens, points_embedding, _ = self.vqvae.encode(batch) # [bs, t//4, self.token_num]
            # print("points_tokens: ", points_tokens.shape)
            # print("points_tokens: ", points_tokens[:, :, 0:5])
            # print("points_tokens: ", points_tokens[:, :, 0:5].contiguous().view(bs, -1))
            # exit()
            # pose_tokens = points_tokens[:, :, 0:5].contiguous()
            # face_tokens = points_tokens[:, :, 5:10].contiguous()
            # rhand_tokens = points_tokens[:, :, 10:15].contiguous()
            # lhand_tokens = points_tokens[:, :, 15:20].contiguous()

        # return pose_tokens, face_tokens, rhand_tokens, lhand_tokens, points_embedding
        return points_tokens, points_embedding


    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        self.vqvae.eval()
        # print("pose: ", batch["pose"].shape)
        # print("pose_len: ", batch["points_len"].long())

        points_tokens, points_embedding = self._points2tokens(batch)
        # print("points_tokens: ", points_tokens.shape, points_tokens[:, 0, :])
        # print("pose_tokens, face_tokens, rhand_tokens, lhand_tokens: ", pose_tokens.shape, face_tokens.shape, rhand_tokens.shape, lhand_tokens.shape)
        # print("pose_tokens, face_tokens, rhand_tokens, lhand_tokens: ", pose_tokens[:, 0, :], face_tokens[:, 0, :], rhand_tokens[:, 0, :], lhand_tokens[:, 0, :])
        # exit()
        
        points_len = batch["points_len"].long()

        word_tokens = batch["tokens"].long()
        bsz, _ = word_tokens.size()
        word_mask = word_tokens.ne(self.text_dict.pad())

        
        word_feat, predicted_lengths_lprobs, word_mask = self.encoder(word_tokens=word_tokens, mask=word_mask)


        # print("word_tokens: ", word_tokens)
        # length loss
        length_target = points_len.unsqueeze(-1)
        assert max(length_target) < predicted_lengths_lprobs.size(-1)
        length_loss = -predicted_lengths_lprobs.gather(dim=-1, index=length_target)
        length_loss = length_loss.sum() / bsz * 0.01

        # pose loss
        bsz, tgt_len, tok_num = points_tokens.size()
        nll_loss = self.enc_dec(points_tokens, points_len, word_feat, word_mask, self.points_pad)

        loss = nll_loss.mean()
        
        for i in range(tok_num-1, -1, -1):
            self.log('{}/nll_loss_{}'.format(mode, i), nll_loss[i].detach(), prog_bar=True)

        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        self.log('{}/length_loss'.format(mode), length_loss.detach(), prog_bar=True)
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

    def enc_dec(self, tgt_tokens, tgt_len, src_feat, src_mask, pad_idx):  
        min_num_masks = 1

        bs, t, tok_num = tgt_tokens.size()

        teacher_len = 4
        tgt_inp = torch.cat([torch.ones(bs, teacher_len, tok_num).long().to(tgt_tokens.device) * self.points_bos, tgt_tokens], dim=1) # [bs, tgt_len+1, tok_num]
        tgt_out = torch.cat([tgt_tokens, torch.ones(bs, teacher_len, tok_num).long().to(tgt_tokens.device) * self.points_pad], dim=1) # [bs, tgt_len+1, tok_num]
        for i in range(bs):
            length = tgt_len[i].cpu().item()
            tgt_inp[i, length+teacher_len:, :] = self.points_pad
            tgt_out[i, length, :] = self.points_eos
            tgt_out[i, length+teacher_len:, :] = self.points_pad

        # print("tgt_tokens: ", tgt_tokens.shape, tgt_tokens[0, :10, 0])
        # print("tgt_inp: ", tgt_inp.shape, tgt_inp[0, :10, 0])
        # print("tgt_out: ", tgt_out.shape, tgt_out[0, :10, 0])
        # print("tgt_len: ", tgt_len, tgt_inp.shape)
        # exit()
        size = max(tgt_len+teacher_len)
        tgt_mask = self._get_mask(tgt_len+teacher_len, size) # [bs, tgt_len]
        
        # print("dec inp: ", tgt_inp.shape, src_feat.shape, src_mask.shape, tgt_mask.shape)
        
        logits = self.decoder(trg_tokens=tgt_inp, encoder_output=src_feat, src_mask=src_mask, trg_mask=tgt_mask, 
                              mask_future=False, window_mask_future=True, window_size=teacher_len)

        # nll loss function
        # lprobs = F.log_softmax(logits, dim=-1)
        # print(lprobs[0, 0, 1, 1024:1034], lprobs[0, 0, 1, 2044:2052])
        # print(lprobs[0, 0, 2, 2048:2058], lprobs[0, 0, 2, 3068:3074])
        # print(lprobs[0, 0, 3, 3072:3083], lprobs[0, 0, 3, 4092:4100])
        # print(lprobs[0, 0, 4, 4096:4106], lprobs[0, 0, 4, -10:])
        # exit()

        # print("logits: ", logits.shape, tgt_out.shape)
        nll_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1), reduction="none")
        nll_loss = nll_loss.view(bs*size, tok_num).sum(0)

        non_pad = tgt_out.ne(self.points_pad) # [bs, tgt_len, tok_num]
        non_pad = non_pad.view(bs*size, tok_num).sum(0)
        nll_loss = nll_loss / non_pad
        
        return nll_loss

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
        word_feat, _, word_mask = self.encoder(word_tokens=word_tokens, mask=word_mask)
        
        points_len = batch["points_len"].long()
        max_len = max(points_len)
        # print("predict_len and real length: ", predict_len, batch["points_len"].long())

        points_pred = self.greedy_dec(word_feat, word_mask, max_len)
        return points_pred, points_len


    def greedy_dec(self, word_feat, word_mask, max_len, teacher_len=4):
        bsz = word_feat.size(0)
        ys = torch.ones(bsz, teacher_len, 20).long().to(word_feat.device) * self.points_bos # [bs, teacher_len, 20]
        for i in range(max_len//4):
            logits = self.decoder(ys, encoder_output=word_feat, src_mask=word_mask, trg_mask=None, 
                                mask_future=False, window_mask_future=True, window_size=teacher_len) # [bs, t, tok_num, vocab_size]
            lprobs = F.log_softmax(logits[:, -4:, :, :], dim=-1) # [bs, 4, tok_num, vocab_size]
            lprobs[:, :, :, -3:] = float("-inf")
            _, preds = torch.max(lprobs, dim = -1) # [bs, 4, tok_num]
            ys = torch.cat([ys, preds], dim=1)
        return ys[:, 4:, :]


    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand, log=False)
        # self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
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



def delay_subsequent_mask(size, teacher_len):
    pos = torch.arange(0, size).unsqueeze(0).repeat(size, 1)
    left = (torch.arange(0, size) - teacher_len).unsqueeze(1)
    mask = (pos <= left).long() + torch.eye(size).long()
    return mask.unsqueeze(0).bool()

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
        self.trg_trg_att = MultiHeadedAttention3D(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention3D(num_heads, size, dropout=dropout)

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

        self.point_tok_embedding = WordEmbeddings(embedding_dim=hidden_size, vocab_size=vocab_size, 
            pad_idx=points_pad, num_heads=8, norm_type=None, activation_type=None, scale=False, scale_factor=None)

        self.spatial_abs_pe = nn.Parameter(torch.randn(1, 1, 20, hidden_size), requires_grad=True)
        self.temporal_abs_pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, self._output_size, bias=False)

        self.register_buffer("subsequen_mask", delay_subsequent_mask(1000, 4))


    def forward(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, tag_name=None):
        """x: trg_embed
        """

        # print("the decoder input is embeded!")
        bsz, tgt_len, tok_num = trg_tokens.size()

        # print("trg_tokens: ", trg_tokens.shape, trg_tokens[0, :, :])
                
        x = self.point_tok_embedding(trg_tokens, trg_mask)
        x = x + self.spatial_abs_pe # [bsz, tgt_len, 20, emb_dim]

        x = x.permute(0, 2, 1, 3).contiguous().view(bsz*tok_num, tgt_len, -1) # [bs*tok_num, tgt_len, emd_dim]
        x = x + self.temporal_abs_pe(x)

        x = x.view(bsz, tok_num, tgt_len, -1)
        
        x = self.emb_dropout(x)

        if mask_future:
            size = x.size(-2)
            if trg_mask is not None:
                # print("subsequent_mask: ", subsequent_mask(x.size(-2)).bool().shape)
                trg_mask = trg_mask.unsqueeze(1) & subsequent_mask(size).bool().to(x.device)
            else:
                trg_mask = subsequent_mask(size).bool().to(x.device)
        
        if window_mask_future:
            assert window_size is not None
            size = x.size(-2)
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & self.subsequen_mask[:, :size, :size].to(x.device)
            else:
                trg_mask = self.subsequen_mask[:, :size, :size].to(x.device)

        # print("dec inp: ", x.shape, encoder_output.shape, src_mask.shape, trg_mask.shape)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)


        x = self.layer_norm(x)
        x = self.output_layer(x) # [bs, tok_num, t, vocab_size]
        x = x.permute(0, 2, 1, 3).contiguous()  # [bs, t, tok_num, vocab_size]


        for i in range(tok_num):
            out_mask = torch.zeros(1, 1, 1, self._output_size).to(x.device)
            out_mask[..., i*1024: (i+1)*1024] = 1
            out_mask[..., -3:] = 1
            x[:, :, i, :] = x[:, :, i, :].masked_fill(~out_mask.bool(), float("-inf"))
        return x
