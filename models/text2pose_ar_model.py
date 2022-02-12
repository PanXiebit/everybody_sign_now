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
from modules.transformer import TransformerEncoder, TransformerDecoder
from modules.transformer.word_embedding import WordEmbeddings
from modules.left_to_right import LeftToRight

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

        self.word_embedding = WordEmbeddings(embedding_dim=512, vocab_size=len(text_dict), pad_idx=text_dict.pad(), num_heads=8, norm_type="batch", activation_type="softsign",)
        self.encoder = TransformerEncoder(hidden_size=512, ff_size=2048, num_heads=8, num_layers=6, dropout=0.3, emb_dropout=0.3)

        self.points_bos = self.vqvae.args.n_codes
        self.points_eos = self.vqvae.args.n_codes + 1
        self.points_pad = self.vqvae.args.n_codes + 2
        vocab_size = self.vqvae.args.n_codes+3
        self.point_tok_embedding = WordEmbeddings(embedding_dim=512, vocab_size=vocab_size, pad_idx=self.points_pad, num_heads=8, norm_type="batch", activation_type="softsign",)
        self.decoder = TransformerDecoder(vocab_size, num_layers=6, num_heads=8, hidden_size=512, ff_size=2048, dropout=0.3, emb_dropout=0.3)

        self.inferece_dec = LeftToRight(text_bos=self.points_bos, 
                                        text_pad=self.points_pad, text_eos=self.points_eos,
                                        text_embedding=self.point_tok_embedding,
                                        decoder=self.decoder,
                                        token_num=self.token_num)
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
        # print("origin points_len: ", batch["points_len"])
        points_len = batch["points_len"].long() * self.token_num // 4
        # print("points_len: ", points_len)
        word_tokens = batch["tokens"].long()

        word_mask = word_tokens.ne(self.text_dict.pad())
        word_embed = self.word_embedding(word_tokens, word_mask)
        # print("word_mask: ", word_mask.shape)
        word_feat = self.encoder(embed_src=word_embed, mask=word_mask)

        # print("after encodr: ", word_feat.shape)  
        
        # add eos
        bs, _ = points_tokens.size()
        pad_tokens = torch.ones((bs, 1), dtype=torch.long).to(points_tokens.device) * self.points_eos
        points_tokens = torch.cat([points_tokens, pad_tokens], dim=-1)
        # print("points_tokens: ", points_tokens.shape)
        for i in range(bs):
            leng = points_len[i]
            points_tokens[i, leng] = self.points_eos
            points_tokens[i, leng+1:] = self.points_pad
        points_inp = points_tokens[:, :-1]
        points_tgt = points_tokens[:, 1:]
        # print("points_tgt, points_inp: ", points_tgt.shape, points_inp.shape)

        # print("points_len: ", points_len, points_inp.shape)
        size = points_inp.size(1)
        points_mask = self._get_mask(points_len + 1, size)
        # exit()

        points_emd = self.point_tok_embedding(points_inp, points_mask)
        logits = self.decoder(trg_embed=points_emd, encoder_output=word_feat, src_mask=word_mask, trg_mask=points_mask)

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
            pose_recon, face_recon = pose_recon[:, :, :vis_len, :], face_recon[:, :, :vis_len, :]
            rhand_recon, lhand_recon = rhand_recon[:, :, :vis_len, :], lhand_recon[:, :, :vis_len, :]
            self.vis("train", "rec_vis", pose_recon, face_recon, rhand_recon, lhand_recon)

            # prediction
            # predictions = torch.argmax(logits, dim=-1)[0:1] # [bs, t*v]
            # print("predictions: ", predictions.shape)
            # predictions_np = predictions.detach().cpu().numpy()
            # idx = np.argwhere(predictions_np[0] >= self.points_eos).tolist()
            # if len(idx) > 0:
            #     # print("idx: ", idx)
            #     idx = idx[0][0]
            #     try:
            #         predictions = predictions[:, :idx]
            #     except:
            #         print("error idx: ", idx) # 不知道在哪里会出现eos，并且出现eos的地方不一定是20的倍数

            # predictions_emb = self.vqvae.codebook.dictionary_lookup(predictions)
            # print("predictions_emb: ", predictions_emb.shape)

            # predictions_emb = predictions_emb.permute(0, 2, 1).contiguous()
            # # print("predictions_emb: ", predictions_emb.shape)

            # predictions_emb = predictions_emb.view(1, 256, -1, 20)
            # # assert predictions_emb.size()[1:] == points_embedding.size()[1:]

            # pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(predictions_emb)
            # pose_pred, face_pred = pose_pred[:, :, :vis_len, :], face_pred[:, :, :vis_len, :]
            # rhand_pred, lhand_pred = rhand_pred[:, :, :vis_len, :], lhand_pred[:, :, :vis_len, :]
            # self.vis("train", "pred_vis", pose_pred, face_pred, rhand_pred, lhand_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

        if batch_idx < 5:
            points_len = batch["points_len"].long()
            vis_len = points_len[0]
            
            pose = self.vqvae.selects(batch["pose"], "pose")[:, :, :vis_len, :]
            face = self.vqvae.selects(batch["face"], "face")[:, :, :vis_len, :]
            rhand = self.vqvae.selects(batch["rhand"], "rhand")[:, :, :vis_len, :]
            lhand = self.vqvae.selects(batch["lhand"], "lhand")[:, :, :vis_len, :]
            self.vis("val", "ori_vis", pose, face, rhand, lhand)

            self.inference(batch)

    
    def inference(self, batch):
        self.vqvae.eval()
        points_tokens, points_embedding = self._points2tokens(batch)
        word_tokens = batch["tokens"].long()

        word_mask = word_tokens.ne(self.text_dict.pad())
        word_embed = self.word_embedding(word_tokens, word_mask)
        word_feat = self.encoder(embed_src=word_embed, mask=word_mask)

        predictions = self.inferece_dec.generate(encoder_out=word_feat,
                                   src_mask=word_mask,
                                   max_output_length=400,
                                   beam_size=5,
                                   alpha=2.0,
                                   n_best=1)

        print("predictions: ", predictions.shape, predictions[:, :10])
        print("reference: ", points_tokens.shape, points_tokens[:, :10])

        predictions = predictions[0:1]
        idx = np.argwhere(predictions[0] == self.points_eos).tolist()
        if len(idx) > 0:
            # print("idx: ", idx)
            idx = idx[0][0]
            try:
                predictions = predictions[:, :idx] # 不知道在哪里会出现eos，并且出现eos的地方不一定是20的倍数
            except:
                print("error idx: ", idx) 

        predictions = torch.from_numpy(predictions).type_as(points_tokens)
        pred_len = predictions.size(-1)
        pred_len = pred_len // 20 * 20
        predictions_emb = self.vqvae.codebook.dictionary_lookup(predictions[:, :pred_len])
        predictions_emb = predictions_emb.permute(0, 2, 1).contiguous()
        predictions_emb = predictions_emb.view(1, 256, -1, 20)
        pose_pred, face_pred, rhand_pred, lhand_pred = self.vqvae.decode(predictions_emb)
        self.vis("val", "pred_vis", pose_pred, face_pred, rhand_pred, lhand_pred)
        
        


    def vis(self, mode, name, pose, face, rhand, lhand):
        mode, name, ori_vis = self.vqvae.visualization(mode, name, pose, face, rhand, lhand)
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
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
