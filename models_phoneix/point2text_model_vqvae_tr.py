from email.policy import default
from turtle import forward
from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from modules.utils import shift_dim
import numpy as np
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.vq_fn import Codebook
import einops
from modules.sp_layer import SPL
from util.plot_videos import draw_frame_2D
from util.wer import get_wer_delsubins
import ctcdecode
from itertools import groupby



class Point2textModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

        # vqvae
        self.points_emb = nn.Linear(150, 512)
        self.enc_vit = Encoder(dim=512, depth=3, heads=8, mlp_dim=2048, dropout = 0.1)
        self.codebook = Codebook(n_codes=args.n_codes, embedding_dim=512)
        self.dec_vit = Encoder(dim=512, depth=3, heads=8, mlp_dim=2048, dropout = 0.1)
        self.dec_linear = nn.Linear(512, 512*3)
        self.pose_spl = SPL(input_size=512, hidden_layers=5, hidden_units=512, joint_size=3, reuse=False, sparse=False, SKELETON="sign_pose")        
        self.hand_spl = SPL(input_size=512, hidden_layers=5, hidden_units=512, joint_size=3, reuse=False, sparse=False, SKELETON="sign_hand")        

        # encoder-decoder

        self.gloss_embedding = nn.Embedding(len(text_dict), 512, text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.pad_id = args.n_codes
        self.eos_id = args.n_codes + 1
        self.points_vocab_size = args.n_codes + 2
        self.points_pad = nn.Parameter(torch.zeros(1, 512))
        self.points_bos = nn.Parameter(torch.zeros(1, 512))
        self.points_eos = nn.Parameter(torch.zeros(1, 512))

        self.pos_emb = PositionalEncoding(0.1, 512, 5000)
        self.transformer = Transformer(emb_dim=512, depth=6, block_size=5000)
        self.out_layer = nn.Linear(512, self.points_vocab_size) 

        # ctc learning
        self.conv = nn.Sequential(nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2), 
                                   nn.BatchNorm1d(512), 
                                   nn.LeakyReLU(),
                                   nn.MaxPool1d(2, 2),
                                   nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm1d(512),
                                   nn.MaxPool1d(2, 2))
        self.ctc_out = nn.Linear(512, len(text_dict))
        self.ctcLoss = nn.CTCLoss(text_dict.blank(), reduction="mean", zero_infinity=True)
        self.decoder_vocab = [chr(x) for x in range(20000, 20000 + len(text_dict))]
        self.decoder = ctcdecode.CTCBeamDecoder(self.decoder_vocab, beam_width=5,
                                                blank_id=text_dict.blank(),
                                                num_processes=10)

        self.save_hyperparameters()

    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        bs, max_len, v = points.size()

        # vqvae
        points_feat = self.points_emb(points) # [bs, max_len, 512]
        points_mask = self._get_mask(skel_len, max_len, points_feat.device)
        points_feat = self.enc_vit(points_feat, points_mask.unsqueeze_(1).unsqueeze_(2)) # [bs, max_len, 512]
        enc_feat = points_feat = einops.rearrange(points_feat, "b t h -> b h t")
        vq_output = self.codebook(points_feat)
        vq_tokens, points_feat, commitment_loss = vq_output['encodings'], vq_output['embeddings'], vq_output["commitment_loss"] # [bs, max_len]
        self.log('{}/commitment_loss'.format(mode), commitment_loss.detach(), prog_bar=True)

        dec_feat = points_feat = einops.rearrange(points_feat, "b h t-> b t h")
        points_feat = self.dec_vit(points_feat, points_mask)  # [b t h]
        
        
        # reconstruction loss
        rec_feat = einops.rearrange(points_feat, "b t h -> (b t) h")
        rec_feat = self.dec_linear(rec_feat)
        rec_feat = rec_feat[points_mask.view(-1)]
        dec_pose = self.pose_spl(rec_feat[:, 0:512])  # [b, h] -> [b, 24]
        dec_rhand = self.hand_spl(rec_feat[:, 512:1024]) # [b, h] -> [b, 63]
        dec_lhand = self.hand_spl(rec_feat[:, 1024:2048]) # [b, h] -> [b, 63]
        dec_points = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1)
        ori_points = einops.rearrange(points, "b t v -> (b t) v")
        ori_points = ori_points[points_mask.view(-1)]
        pose = ori_points[:, :24]
        rhand = ori_points[:, 24:24+63]
        lhand = ori_points[:, 87:150]
        rec_loss = torch.abs(dec_points - ori_points).mean()
        self.log('{}/rec_loss'.format(mode), rec_loss.detach(), prog_bar=True)
        if mode == "train" and self.global_step % 500 == 0:
            self.vis(dec_pose, dec_rhand, dec_lhand, mode, "recons", vis_len=skel_len[0].item())
            self.vis(pose, rhand, lhand, mode, "origin", vis_len=skel_len[0].item())
            self.vis_token(vq_tokens[0, :skel_len[0].item()], mode, "rec")
        elif mode == "val":
            self.vis(dec_pose, dec_rhand, dec_lhand, mode, "recons", vis_len=skel_len[0].item())
            self.vis(pose, rhand, lhand, mode, "origin", vis_len=skel_len[0].item())
            self.vis_token(vq_tokens[0, :skel_len[0].item()], mode, "rec")

        # ctc loss
        # ctc_feat = einops.rearrange(enc_feat, "b t h -> b h t")
        ctc_feat = self.conv(enc_feat)
        ctc_feat = einops.rearrange(ctc_feat, "b h t -> b t h")
        ctc_skel_len = skel_len // 4 
        logits = self.ctc_out(ctc_feat)  # [bs, t, vocab_size]
        lprobs = logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        ctc_loss = self.ctcLoss(lprobs.cpu(), gloss_id.cpu(), ctc_skel_len.cpu(), gloss_len.cpu()).to(lprobs.device)
        self.log('{}/ctc_loss'.format(mode), ctc_loss.detach(), prog_bar=True)


        # transformer loss
        if self.current_epoch >= 20:
            src_feat = self.gloss_embedding(gloss_id)
            src_feat = self.pos_emb(src_feat)
            src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

            tgt_emb_inp = torch.cat([self.points_eos.unsqueeze(1).repeat(bs, 1, 1), dec_feat], dim=1) # [bs, max_len+1, emb_dim]
            tgt_emb_inp = self.pos_emb(tgt_emb_inp)
            
            tgt_out = vq_tokens.new(bs, max_len + 1).fill_(self.pad_id)
            for b in range(bs):
                tgt_out[b, :skel_len[b].item()] = vq_tokens[b, :skel_len[b].item()]
                tgt_out[b, skel_len[b].item()] = self.eos_id
            
            tgt_mask = self._get_mask(skel_len+1, max_len+1, points_feat.device).unsqueeze_(1).unsqueeze_(2)

            out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask)
            out_logits = self.out_layer(out_feat) 

            out_logits = einops.rearrange(out_logits, "b t v -> (b t) v")
            tgt_out = tgt_out.view(-1)
            ce_loss = F.cross_entropy(out_logits, tgt_out, ignore_index=self.pad_id, reduction="none")
            ce_loss = ce_loss.sum() / tgt_mask.sum()
            self.log('{}/ce_loss'.format(mode), ce_loss.detach(), prog_bar=True)

            # total loss
            loss = ctc_loss + commitment_loss + rec_loss + ce_loss
        else:
            loss = ctc_loss + commitment_loss + rec_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        
        return loss, logits, vq_tokens


    def training_step(self, batch):
        loss, _, _ = self.forward(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx <= 2:
            self.generate(batch, batch_idx, total_seq_len=200)
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        points = batch["skel_3d"]
        skel_len = batch["skel_len"]
        
        _, logits, vq_tokens = self.forward(batch, "val") # [bs, t, vocab_size]
        bs = skel_len.size(0)
        
        # TODO! recognition prediction and compute wer
        gloss_logits = F.softmax(logits, dim=-1)
        skel_len = skel_len // 4
        pred_seq, _, _, out_seq_len = self.decoder.decode(gloss_logits, skel_len)


        err_delsubins = np.zeros([4])
        count = 0
        correct = 0
        for i, length in enumerate(gloss_len):
            ref = gloss_id[i][:length].tolist()[:-1]
            hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())][:-1]
            # ref_sent = clean_phoenix_2014(self.text_dict.deocde_list(ref))
            # hyp_sent = clean_phoenix_2014(self.text_dict.deocde_list(hyp))
            # hyp = ref
            # decoded_dict[vname[i]] = (ref, hyp)
            correct += int(ref == hyp)
            err = get_wer_delsubins(ref, hyp)
            err_delsubins += np.array(err)
            count += 1
        return dict(wer=err_delsubins, correct=correct, count=count, vq_tokens=vq_tokens, gloss_id=gloss_id)


    def validation_epoch_end(self, outputs) -> None:
        val_err, val_correct, val_count = np.zeros([4]), 0, 0
        for out in outputs:
            val_err += out["wer"]
            val_correct += out["correct"]
            val_count += out["count"]
            vq_tokens = out["vq_tokens"]
            gloss_id = out["gloss_id"]

        self.log('{}/acc'.format("val"), val_correct / val_count, prog_bar=True)
        self.log('{}_wer'.format("val"), val_err[0] / val_count, prog_bar=True)
        self.log('{}/sub'.format("val"), val_err[1] / val_count, prog_bar=True)
        self.log('{}/ins'.format("val"), val_err[2] / val_count, prog_bar=True)
        self.log('{}/del'.format("val"), val_err[3] / val_count, prog_bar=True)


    @torch.no_grad()
    def generate(self, batch, batch_idx, total_seq_len=300, temperature = 1., default_batch_size = 1, ):
        gloss = batch["gloss_id"]
        bs = gloss.size(0)
        for i in range(bs):
            gloss_id = gloss[i:i+1]
            src_feat = self.gloss_embedding(gloss_id)
            src_feat = self.pos_emb(src_feat)
            src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

            res_emb = self.points_bos.unsqueeze(0)# [1, 1, emb_dim]
            res_emb = self.pos_emb(res_emb, step=0)

            tgt_mask = None
            for step in range(1, total_seq_len):
                out_feat = self.transformer(src_feat, src_mask, res_emb, tgt_mask)
                logits = self.out_layer(out_feat)[:, -1, :] # [1, vocab_size]
                sampled_idx = torch.argmax(logits, dim=-1) # [1, 1]
                if (sampled_idx == self.eos_id).any(): break
                logits = logits[:, :-2]
                predicts = F.gumbel_softmax(logits, tau=0.1, hard=False) # [1, n_codes]
                cur_emb = torch.matmul(predicts, self.codebook.embeddings) # [1, emb_dim]
                cur_emb = self.pos_emb(cur_emb, step=step)
                res_emb = torch.cat([res_emb, cur_emb.unsqueeze(0)], dim=-2)  # [1, t, emb_dim]
                
            pred_feat = res_emb[:, 1:, :] # [1, t, emb_dim]
            pred_feat = einops.rearrange(pred_feat, "b t h -> (b t) h")
            pred_feat = self.dec_linear(pred_feat)
            pred_pose = self.pose_spl(pred_feat[:, 0:512])  # [b, h] -> [b, 24]
            pred_rhand = self.hand_spl(pred_feat[:, 512:1024]) # [b, h] -> [b, 63]
            pred_lhand = self.hand_spl(pred_feat[:, 1024:2048]) # [b, h] -> [b, 63]

            pred_points = torch.cat([pred_pose, pred_rhand, pred_lhand], dim=-1).detach().cpu().numpy()
            # points: [bs, 150]
            show_img = []
            for j in range(pred_points.shape[0]):
                frame_joints = pred_points[j]
                frame = np.ones((256, 256, 3), np.uint8) * 255
                frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
                # Draw the frame given 2D joints
                im = draw_frame_2D(frame, frame_joints_2d)
                show_img.append(im)
            show_img = np.concatenate(show_img, axis=1) # [h, w, c]
            cv2.imwrite("/Dataset/everybody_sign_now_experiments/predictions/epoch={}_batch={}_idx={}.png".format(self.current_epoch, batch_idx, i), show_img)


    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96, last_epoch=-1)
        return [self.optimizer], [scheduler]
    
    def vis_token(self, pred_tokens, mode, name):
        pred_tokens = " ".join([str(x) for x in pred_tokens.cpu().numpy().tolist()])
        self.logger.experiment.add_text("{}/{}_points".format(mode, name), pred_tokens, self.current_epoch)


    def vis(self, pose, rhand, lhand, mode, name, vis_len):
        points = torch.cat([pose, rhand, lhand], dim=-1).detach().cpu().numpy()
        # points: [bs, 150]
        show_img = []
        for j in range(vis_len):
            frame_joints = points[j]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d)
            show_img.append(im)
        show_img = np.concatenate(show_img, axis=1) # [h, w, c]
        show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
        show_img = torchvision.utils.make_grid(show_img, )
        self.logger.experiment.add_image("{}/{}".format(mode, name), show_img, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pose_vqvae', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=512)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
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


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(1)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

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

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        self.self_attn = Attention(heads, dim)
        self.cross_attn = Attention(heads, dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = BertLayerNorm(dim)
        self.norm2  = BertLayerNorm(dim)
        self.norm3  = BertLayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, enc_out, tgt_mask, src_mask):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, pad_future_mask, enc_out, src_mask):
        for layer in self.layers:
            x = layer(x, enc_out, pad_future_mask, src_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim=512, depth=6, block_size=2000):
        super().__init__()
        casual_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("casual_mask", casual_mask.bool().view(1, 1, block_size, block_size))

        self.encoder = Encoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)
        self.decoder = Decoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)


    def forward(self, src_feat, src_mask, tgt_feat, tgt_mask): 
        
        enc_out = self.encoder(src_feat, src_mask)
        bs, t, _ = tgt_feat.size()
        casual_mask = self.casual_mask[:, :, :t, :t]
        if tgt_mask is not None:
            pad_future_mask = casual_mask & tgt_mask
        else:
            pad_future_mask = casual_mask
        dec_out = self.decoder(tgt_feat, pad_future_mask, enc_out, src_mask)
        return dec_out


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