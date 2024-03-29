import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torchvision
from torchvision.utils import save_image
from modules.mask_predict import MaskPredict
from modules.transformer import TransformerEncoder, TransformerDecoder
import random
from modules.transformer.utils import BertLayerNorm


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict, seed=888):
        super().__init__()
        self.args = args
        self.text_dict = text_dict
        
        self.token_num = 20
        self.max_target_positions = args.max_frames_num + 1
        self.max_source_positions = 500

        # Load VQ-VAE and set all parameters to no grad
        from .pose_vqvae_vit_model import PoseVitVQVAE
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
        self.encoder = TransformerEncoder(text_dict, max_source_positions, max_target_positions, 
            hidden_size=512, ff_size=2048, num_heads=8, num_layers=6, dropout=0.1, emb_dropout=0.1)

        # decoder
        self.points_mask = self.vqvae.args.n_codes
        self.points_pad = self.vqvae.args.n_codes + 1
        vocab_size = self.vqvae.args.n_codes + 2
        self.decoder = TransformerDecoder(vocab_size, max_target_positions * self.token_num // 4, 
            self.points_pad, num_layers=6, num_heads=8, hidden_size=512, ff_size=2048, dropout=0.1, emb_dropout=0.1)

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
        points_tokens, points_embedding = self.vqvae.encode(batch) # [bs*t, 1, self.token_num]
        points_tokens = points_tokens.view(bs, t*self.token_num // 4)
        return points_tokens, points_embedding

    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        self.vqvae.eval()
        points_tokens, points_embedding = self._points2tokens(batch)
        points_len = batch["points_len"].long() * self.token_num // 4

        # print("points_len: ", batch["points_len"].long(), points_len)
        word_tokens = batch["tokens"].long()
        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat, predicted_lengths_lprobs = self.encoder(word_tokens=word_tokens, mask=word_mask)

        min_num_masks = 1
        bs, _ = points_tokens.size()
        points_inp = points_tokens.clone()
        points_tgt_cp = points_tokens.clone()
        points_tgt = torch.ones_like(points_tokens).to(points_tokens.device) * self.points_pad

        for i in range(bs):
            length = points_len[i].cpu().item()
            points_tokens[i, length:] = self.points_pad

            sample_size = self.random.randint(min_num_masks, length)
            ind = self.random.choice(length, size=sample_size, replace=False)
            points_inp[i, ind] = self.points_mask
            points_tgt[i, ind] = points_tgt_cp[i, ind]
        
        # print("points_tokens: ", points_tokens.shape, points_tokens[:2,:10])
        # print("points_inp: ", points_inp.shape, points_inp[:2, :10])
        # print("points_tgt: ", points_tgt.shape, points_tgt[:2, :10])
        
        # print("points_len: ", points_len, points_inp.shape)
        size = points_inp.size(1)
        points_mask = self._get_mask(points_len, size)
        logits = self.decoder(trg_tokens=points_inp, encoder_output=word_feat, src_mask=word_mask, trg_mask=points_mask, 
                              mask_future=False, window_mask_future=True, window_size=self.token_num)

        # nll loss function
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1)) 
        target = points_tgt.view(-1, 1)
        non_pad_mask = target.ne(self.points_pad)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        # length loss 
        length_target = batch["points_len"].long().unsqueeze(-1)
        assert max(length_target) < predicted_lengths_lprobs.size(-1)
        length_loss = -predicted_lengths_lprobs.gather(dim=-1, index=length_target)

        reduce = True
        if reduce:
            tokens_nums = non_pad_mask.sum()
            assert tokens_nums != 0
            nll_loss = nll_loss.sum() / tokens_nums
            smooth_loss = smooth_loss.sum() / tokens_nums
            length_loss = length_loss.sum() / tokens_nums   # TODO!

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss + length_loss

        non_pad_ratio = non_pad_mask.sum() / points_len.sum()
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        self.log('{}/nll_loss'.format(mode), nll_loss.detach(), prog_bar=True)
        self.log('{}/smooth_loss'.format(mode), smooth_loss.detach(), prog_bar=True)
        self.log('{}/length_loss'.format(mode), length_loss.detach(), prog_bar=True)
        self.log('{}/non_pad_ratio'.format(mode), non_pad_ratio.detach(), prog_bar=True)
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
        points_tokens, points_embedding = self._points2tokens(batch)
        # print("gt tokens: ", points_tokens[:, :15])

        vis_len = batch["points_len"].long()[0]
        pose = self.vqvae.selects(batch["pose"], "pose")[:, :, :vis_len, :]
        face = self.vqvae.selects(batch["face"], "face")[:, :, :vis_len, :]
        rhand = self.vqvae.selects(batch["rhand"], "rhand")[:, :, :vis_len, :]
        lhand = self.vqvae.selects(batch["lhand"], "lhand")[:, :, :vis_len, :]
        self.vis("val", "ori_vis", pose, face, rhand, lhand)

        word_tokens = batch["tokens"].long()
        bsz = word_tokens.size(0)
        word_mask = word_tokens.ne(self.text_dict.pad())
        # print("word_tokens: ", word_tokens[:, :10])
        word_feat, predicted_lengths_probs = self.encoder(word_tokens=word_tokens, mask=word_mask)
        predict_len = torch.argmax(predicted_lengths_probs, dim=-1) # [bs]
        predict_len[predict_len < 2] = 2

        predict_len = predict_len
        # print("predict_len and real len: ", predict_len, batch["points_len"].long())
        max_len = predict_len.max().item()
        past_self = None
        past_points_mask = None
        res = None

        # test
        for pos in range(1, predict_len[0] + 1):
            print("\n")
            print("="*10 + "pos {}: ".format(pos))
            if pos > 5:
                exit()
            step_len = self.token_num

            points_mask = predict_len.new(bsz, step_len).fill_(1)
            points_mask = (points_mask * (pos <= predict_len).unsqueeze_(1)).bool()

            if past_points_mask is None:
                past_points_mask = points_mask
            else:
                past_points_mask = torch.cat([past_points_mask, points_mask], dim =-1)

            tgt_tokens = word_tokens.new(bsz, step_len).fill_(self.points_mask)
            tgt_tokens = (1 - points_mask.long()) * tgt_tokens + points_mask.long() * self.points_pad

            tgt_tokens, present_self = self.decoding_strategy.generate(self, tgt_tokens, word_feat, word_mask, points_mask, past_points_mask, past_self, self.points_pad, self.points_mask)

            if past_self == None:
                past_self = present_self
            else:
                past_self = torch.cat([past_self, present_self], dim=-2) 

            # print("step, past_self: ", past_self.shape)

            if res is None:
                res = tgt_tokens
            else:
                res = torch.cat([res, tgt_tokens], dim=-1)
        
        for idx in range(bsz):
            prediction_vis = res[0:1, :predict_len[idx] * self.token_num]
            if not (prediction_vis >= self.points_mask).any():
                predictions_emb = self.vqvae.codebook.dictionary_lookup(prediction_vis)
                predictions_emb = predictions_emb.permute(0, 2, 1).contiguous()
                predictions_emb = predictions_emb.view(1, 256, -1, 20)
                
                pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(predictions_emb)
                self.vis("val", "pred_vis_{}".format(idx), pose_pred, face_pred, rhand_pred, lhand_pred)
        return res


    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand)
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


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
