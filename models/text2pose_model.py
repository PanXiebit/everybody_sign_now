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


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, ):
        super().__init__()
        self.args = args
        # Load VQ-VAE and set all parameters to no grad
        from .pose_vqvae_single_vit_model import PoseSingleVQVAE
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  PoseSingleVQVAE.load_from_checkpoint(args.pose_vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        self.save_hyperparameters()


    @torch.no_grad()
    def encode_to_z(self, x):
        z_indices = self.vqvae.encode(x, include_embeddings=False)
        z_indices = z_indices.view(z_indices.shape[0], -1)
        return z_indices

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def get_reconstruction(self, x):
        x_recons = self.vqvae.get_reconstruction(x)
        return x_recons


    def forward(self, x, c, label):

        z_indices = self.encode_to_z(x)
        

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch['video']      # [bs, c, t, h, w]
        label = batch["label"]

        if self.use_frame_cond:
            x = x[:, :, self.n_cond_frames:, :, :]
            c = x[:, :, :self.n_cond_frames, :, :]

        logits, target = self(x, c, label)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)
        with torch.no_grad():
            if batch_idx <= 2:
                video = self.inference(batch, steps=4096)
                grid = torchvision.utils.make_grid(video[0])
                self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pose_vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
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
