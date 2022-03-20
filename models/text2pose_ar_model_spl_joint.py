import os
import itertools
import einops
import numpy as np
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
from modules.transformer.word_embedding import WordEmbeddings
from modules.left_to_right import LeftToRight
from modules.transformer.multihead_attention import MultiHeadedAttention
from modules.transformer.utils import gelu, BertLayerNorm, GeLU
from modules.transformer.position_encoding import PositionalEncoding
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.cross_entropy import CandidatePenaltyCrossEntropyCriterion, CrossEntropyCriterionWCustomMetrics


class Text2PoseModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()
        self.args = args
        self.text_dict = text_dict

        # Load VQ-VAE and set all parameters to no grad
        from .pose_vqvae_vit_model_spl_seperate import PoseVitVQVAE
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  PoseVitVQVAE.load_from_checkpoint(args.pose_vqvae, hparams_file=args.hparams_file)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        self.word_embedding = WordEmbeddings(embedding_dim=512, vocab_size=len(text_dict), pad_idx=text_dict.pad(), num_heads=8, norm_type="batch", activation_type="softsign",)
    
        self.encoder = TransformerEncoder(vocab_size=len(text_dict), pad_idx=text_dict.pad(), hidden_size=512, ff_size=2048, num_heads=8, num_layers=6, dropout=0.3, emb_dropout=0.3)

        self.points_bos = self.vqvae.args.n_codes
        self.points_eos = self.vqvae.args.n_codes + 1
        self.points_pad = self.vqvae.args.n_codes + 2
        vocab_size = self.vqvae.args.n_codes+3
        self.decoder = TransformerDecoder(vocab_size, self.points_pad, num_layers=6, num_heads=8, hidden_size=512, ff_size=2048, dropout=0.3, emb_dropout=0.3)

        self.penalty = True
        if self.penalty:
            self.criterion = CandidatePenaltyCrossEntropyCriterion(self.points_pad, rank_alpha=0.3, candidate_type="prev_context")
        else:
            self.criterion = CrossEntropyCriterionWCustomMetrics(self.points_pad)
        self.save_hyperparameters()




    def _get_mask(self, x_len, size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

        
    def forward(self, batch, mode):
        

        self.vqvae.eval()
        with torch.no_grad():
            pose = batch["pose"][..., [1,0,2,3,4,5,6,7]] # [bs, c, t, v]
            rhand = batch["rhand"]
            lhand = batch["lhand"]
            bs, _, t, _ = pose.size()

            vq_pose = einops.rearrange(pose, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_rhand = einops.rearrange(rhand, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_lhand = einops.rearrange(lhand, "b c t v -> (b t) c v").unsqueeze(-2)
            # print(pose.shape, rhand.shape, lhand.shape)
            points_tokens, points_embedding, _ = self.vqvae.encode(vq_pose, vq_rhand, vq_lhand) # [bs*t, 3]
            # print("points_tokens: ", points_tokens.shape) 
            points_tokens = einops.rearrange(points_tokens, "(b t) n -> b (t n)", b=bs, n=3)



        points_len = batch["points_len"].long()
        word_tokens = batch["tokens"].long()
        # print("points_len: ", points_len)
        # print("word_tokens: ", word_tokens.shape, word_tokens[:3, :])
        # print("points_tokens: ", points_tokens.shape, points_tokens[:3, :])

        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat = self.encoder(word_tokens=word_tokens, mask=word_mask)

        # # add eos
        # bs, _ = points_tokens.size()
        # pad_tokens = torch.ones((bs, 1), dtype=torch.long).to(points_tokens.device) * self.points_eos
        # bos_tokens = torch.ones((bs, 1), dtype=torch.long).to(points_tokens.device) * self.points_bos
        # points_tokens = torch.cat([bos_tokens, points_tokens, pad_tokens], dim=-1)
        # # print("points_tokens: ", points_tokens.shape)

        # for i in range(bs):
        #     leng = points_len[i]
        #     points_tokens[i, leng] = self.points_eos
        #     points_tokens[i, leng+1:] = self.points_pad
        # points_inp = points_tokens.clone()[:, :-1].contiguous()
        # points_tgt = points_tokens.clone()[:, 1:].contiguous()
        # print("is contiguous: ", points_tokens.is_contiguous(), points_tgt.is_contiguous(), points_inp.is_contiguous())
        # exit()

        size = max(points_len)
        points_mask = self._get_mask(points_len + 1, size)

        # with torch.autograd.set_detect_anomaly(True):
        logits = self.decoder(trg_tokens=points_tokens, encoder_output=word_feat, src_mask=word_mask, trg_mask=points_mask)
        if self.penalty:
            loss, mle_loss, custom_loss = self.criterion(logits, points_tgt)
            mle_loss = mle_loss.sum() / points_mask.sum()
            custom_loss = custom_loss.sum() / points_mask.sum()
            self.log('{}/mle_loss'.format(mode), mle_loss.detach(), prog_bar=True)
            self.log('{}/custom_loss'.format(mode), custom_loss.detach(), prog_bar=True)
        else:
            loss = self.criterion(logits, points_tgt)
        

        loss = loss.sum() / points_mask.sum()
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        

        with torch.no_grad():
            if self.global_step % 500 == 0:
                # original
                pred_logits = logits.clone()
                
                pred_logits[:, :, -3:] = float("-inf")
                pred = torch.argmax(pred_logits, dim=-1)[:, :-1]
                assert (pred < self.vqvae.args.n_codes).all()

                pred_emb = self.vqvae.codebook.dictionary_lookup(pred)
                pred_emb = einops.rearrange(pred_emb, "b t h -> b h t")
                pred_pose, pred_rhand, pred_lhand = self.vqvae.decode(pred_emb)
                
                pose = pose[..., [1,0,2,3,4,5,6,7,]]
                pred_pose = pred_pose[..., [1,0,2,3,4,5,6,7,]]
                for i in range(4):
                    length = min(points_len[i] // 3, 16)
                    ori_vis = self.visualization(pose[i, :, :length, :], rhand[i, :, :length, :], lhand[i, :, :length, :])
                    pred_vis = self.visualization(pred_pose[i, :, :length, :], pred_rhand[i, :, :length, :], pred_lhand[i, :, :length, :])
                    
                    vis = torch.cat([ori_vis, pred_vis], dim=-2) # [t, 3, 2h ,w]
                    vis = torchvision.utils.make_grid(vis, nrow=16)
                    self.logger.experiment.add_image("{}/vis_{}".format(mode, i), vis, self.global_step)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.forward(batch, "val")
        if batch_idx < 2:
            self.inference(batch, batch_idx)

    def inference(self, batch, batch_idx, max_len=201):
        self.vqvae.eval()

        with torch.no_grad():
            pose = batch["pose"][..., [1,0,2,3,4,5,6,7]] # [bs, c, t, v]
            rhand = batch["rhand"]
            lhand = batch["lhand"]
            bs, _, t, _ = pose.size()
            vq_pose = einops.rearrange(pose, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_rhand = einops.rearrange(rhand, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_lhand = einops.rearrange(lhand, "b c t v -> (b t) c v").unsqueeze(-2)
            points_tokens, points_embedding, _ = self.vqvae.encode(vq_pose, vq_rhand, vq_lhand) # [bs*t, 3]
            points_tokens = einops.rearrange(points_tokens, "(b t) n -> b (t n)", b=bs)

        sents = batch["sents"]
        points_len = batch["points_len"].long() * 3
        word_tokens = batch["tokens"].long()
        # print("points_len: ", points_len)
        # print("word_tokens: ", word_tokens.shape, word_tokens[:3, :])
        # print("points_tokens: ", points_tokens.shape, points_tokens[:3, :])

        word_mask = word_tokens.ne(self.text_dict.pad())
        word_feat = self.encoder(word_tokens=word_tokens, mask=word_mask)

        bsz = word_feat.size(0)
        ys = torch.ones(bsz, 1).long().to(word_feat.device) * self.points_bos # [bs, 1]

        for i in range(max_len):
            logits = self.decoder(ys, encoder_output=word_feat, src_mask=word_mask, trg_mask=None) # [bs, t, vocab_size]
            lprobs = F.log_softmax(logits[:, -1, :], dim=-1) # [bs, vocab_size]
            if i%3 != 0: lprobs[:, -3:] = float("-inf")
            _, preds = torch.max(lprobs, dim = -1, keepdim=True) # [bs, 1]
            if (preds == self.points_eos).all(): break
            ys = torch.cat([ys, preds], dim=1)
        res = ys[:, 1:]
        for i in range(4):
            pred_sent = " ".join([str(w) for w in res[i].cpu().numpy().tolist()])
            self.logger.experiment.add_text("words_{}".format(i + batch_idx), sents[i] + "\n" + pred_sent, self.current_epoch)
        return 

    def visualization(self, pose, rhand, lhand):
        # visualize
        ori_vis = []

        c, t, v = pose.size()
        pose = pose.permute(1, 2, 0).contiguous()  # [t, v, c]
        rhand = rhand.permute(1, 2, 0).contiguous()
        lhand = lhand.permute(1, 2, 0).contiguous()

        for i in range(pose.size(0)):   
            pose_anchor = (640, 360)
            pose_list = self._tensor2numpy(pose[i], pose_anchor, "pose", 25, list(range(8))) # [3V]


            rhand_list = self._tensor2numpy(rhand[i], pose_anchor, "rhand", 21, list(range(21)))# , rhand_anchor[0] * 640, rhand_anchor[1] * 360)
            lhand_list = self._tensor2numpy(lhand[i], pose_anchor, "lhand", 21, list(range(21))) # , lhand_anchor[0] * 640, lhand_anchor[1] * 360)

            canvas = self._render(pose_list, rhand_list, lhand_list)
            canvas = torch.FloatTensor(canvas) # [h, w, c]
            canvas = canvas.permute(2, 0, 1).contiguous().unsqueeze(0)
            
            ori_vis.append(canvas) # [1, c, h, w]
        ori_vis = torch.cat(ori_vis, dim=0)
        return ori_vis
        # ori_vis = torchvision.utils.make_grid(ori_vis, )
        # self.logger.experiment.add_image("{}/{}_{}".format(mode, name, idx), ori_vis, self.global_step)
        # return mode, name, ori_vis
    
    
    def _tensor2numpy(self, points, anchor, part_name, keypoint_num, pose_tokens):
        """[v, c]]
        """
        points = points.detach().cpu().numpy()
        v, c = points.shape
        # [[17, 15, 0, 16, 18], [0, 1, 8, 9, 12], [4, 3, 2, 1, 5], [2, 1, 5, 6, 7]]
        # pose_tokens = []
        assert points.shape[0] == len(pose_tokens)

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

        # img = Image.fromarray(canvas[:, :, [2,1,0]])

        return canvas # [256, 256, 3]

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
    def __init__(self, vocab_size, pad_idx, hidden_size, ff_size, num_heads, num_layers, dropout, emb_dropout):
        super(TransformerEncoder, self).__init__()
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

        self.abs_pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)


    def forward(self, word_tokens, mask):
        """
        """
        if word_tokens.ndim == 2:
            x = self.word_embedding(word_tokens, mask)
        elif word_tokens.ndim == 3:
            x = word_tokens
        else:
            raise ValueError("word_token dim is not 2 or 3!")
            
        x = x + self.abs_pe(word_tokens) 

        x = self.emb_dropout(x)  # [bs, length, embed_size]

        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x

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


        self.point_tok_embedding = WordEmbeddings(embedding_dim=hidden_size, vocab_size=vocab_size, 
            pad_idx=points_pad, num_heads=8, norm_type=None, activation_type=None, scale=False, scale_factor=None)

        self.conv1 = nn.Conv1d(512, 512, kernel_size=3, stride=3)
        self.norm1 = nn.LayerNorm(512)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=4)
        self.norm2 = nn.LayerNorm(512)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, self._output_size, bias=False)



    def forward(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True):
        x = self.point_tok_embedding(trg_tokens, trg_mask)
        
        print("x: ", x.shape)
        exit()
        x = x + self.abs_pe(x)
        x = self.emb_dropout(x)

        if mask_future:
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & subsequent_mask(x.size(1)).bool().to(x.device)
            else:
                trg_mask = subsequent_mask(x.size(1)).bool().to(x.device)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x