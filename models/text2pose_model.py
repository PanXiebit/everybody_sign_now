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
from x_transformers import TransformerWrapper, Encoder, Decoder


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()
        self.args = args
        self.text_dict = text_dict

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
        self.token_num = 20


        self.transformer_enc = TransformerWrapper(
            num_tokens = len(text_dict), 
            max_seq_len = 512,
            attn_layers = Encoder(
                dim = 512,
                depth = 6,
                heads = 8))

        self.points_eos = self.vqvae.args.n_codes
        self.points_pad = self.vqvae.args.n_codes + 1

        self.transformer_dec = TransformerWrapper(
            num_tokens = self.vqvae.args.n_codes + 2,
            max_seq_len = 300 * self.token_num, # TODO?
            attn_layers = Decoder(
                dim = 512,
                depth = 6,
                heads = 8,
                attn_dim_head = 64,
                cross_attend=True))

        self.save_hyperparameters()


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

        
    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        points_tokens, points_embedding = self._points2tokens(batch)
        points_len = batch["points_len"].long() * self.token_num // 4
        word_tokens = batch["tokens"].long()

        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat = self.transformer_enc(word_tokens, mask=word_mask, return_embeddings=True)
        
        # add eos
        bs, _ = points_tokens.size()
        pad_tokens = torch.ones((bs, 1), dtype=torch.long).to(points_tokens.device) * self.points_eos
        points_tokens = torch.cat([points_tokens, pad_tokens], dim=-1)
        for i in range(bs):
            leng = points_len[i]
            points_tokens[i, leng] = self.points_eos
            points_tokens[i, leng+1:] = self.points_pad
        points_inp = points_tokens[:, :-1]
        points_tgt = points_tokens[:, 1:]

        size = points_inp.size(1)
        points_mask = self._get_mask(points_len, size)
        
        attn_mask = torch.triu(torch.ones((size, size), dtype=torch.long), diagonal=1) == 0
        attn_mask = attn_mask.to(points_inp.device)

        logits = self.transformer_dec(x=points_inp, 
                                      mask=points_mask,
                                      attn_mask=attn_mask,
                                      context=word_feat,
                                      context_mask=word_mask)
        
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), points_tgt.reshape(-1), reduction='none')
        
        loss = (loss * points_mask.view(-1)).sum() / points_mask.sum()
        self.log('train/loss', loss, prog_bar=True)

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
            self.vis("train", "rec_vis", pose_recon, face_recon, rhand_recon, lhand_recon)

            # prediction
            predictions = torch.argmax(logits, dim=-1)
            predictions = self.vqvae.codebook.dictionary_lookup(predictions)
            pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(points_embedding)
            self.vis("train", "pred_vis", pose_pred, face_pred, rhand_pred, lhand_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

        if batch_idx < 10:
            points_len = batch["points_len"].long()
            vis_len = points_len[0]
            
            pose = self.vqvae.selects(batch["pose"], "pose")[:, :, :vis_len, :]
            face = self.vqvae.selects(batch["face"], "face")[:, :, :vis_len, :]
            rhand = self.vqvae.selects(batch["rhand"], "rhand")[:, :, :vis_len, :]
            lhand = self.vqvae.selects(batch["lhand"], "lhand")[:, :, :vis_len, :]
            self.vis("val", "ori_vis", pose, face, rhand, lhand)

    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand)
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
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
