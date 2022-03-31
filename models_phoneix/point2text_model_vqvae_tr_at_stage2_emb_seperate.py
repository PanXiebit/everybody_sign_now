from email.policy import default
from turtle import forward
from matplotlib.pyplot import axis, text
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
from modules.mask_strategy import *
from util.dtw import calculate_dtw, dtw



class Point2textModelStage2(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

        # vqvae
        from .point2text_model_vqvae_tr_nat_stage1_seperate import Point2textModel
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  Point2textModel.load_from_checkpoint(args.pose_vqvae, hparams_file=args.vqvae_hparams_file)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # encoder-decoder
        self.gloss_embedding = nn.Embedding(len(text_dict), 512, text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        self.bos_id = self.vqvae.args.n_codes
        self.eos_id = self.vqvae.args.n_codes + 1
        self.pad_id = self.vqvae.args.n_codes + 2
        self.points_vocab_size = self.vqvae.args.n_codes + 3
        self.point_embedding = nn.Embedding(self.points_vocab_size, 512, self.pad_id)
        self.point_embedding.weight.data.normal_(mean=0.0, std=0.02)


        self.pos_emb = PositionalEncoding(0.1, 512, 5000)
        self.transformer = Transformer(emb_dim=512, depth=6, block_size=5000)
        self.out_layer = nn.Linear(512, self.points_vocab_size) 

        from .point2text_model_cnn import BackTranslateModel as BackTranslateModel1
        from .point2text_model import BackTranslateModel as BackTranslateModel2
        self.back_translate_model1 = BackTranslateModel1.load_from_checkpoint(args.backmodel, hparams_file=args.backmodel_hparams_file)
        self.back_translate_model2 = BackTranslateModel2.load_from_checkpoint(args.backmodel2, hparams_file=args.backmodel_hparams_file2)

        self.random = np.random.RandomState(1234)
        self.save_hyperparameters()


    def compute_ctc(self, dec_feat, skel_len, gloss_id, gloss_len):
        ctc_feat = self.conv(dec_feat)
        ctc_feat = einops.rearrange(ctc_feat, "b h t -> b t h")
        ctc_skel_len = skel_len // 4 
        ctc_logits = self.ctc_out(ctc_feat)  # [bs, t, vocab_size]
        lprobs = ctc_logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        ctc_loss = self.ctcLoss(lprobs.cpu(), gloss_id.cpu(), ctc_skel_len.cpu(), gloss_len.cpu()).to(lprobs.device)
        return ctc_logits, ctc_loss # [b t v], [t b v]

    def compute_seq2seq_ce(self, gloss_id, vq_tokens, skel_len):
        """vq_tokens: [bs, t]
        """
        skel_len = skel_len * 3
        max_len = max(skel_len)
        bs = vq_tokens.size(0)

        src_feat = self.gloss_embedding(gloss_id)
        src_feat = self.pos_emb(src_feat)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        tgt_inp = vq_tokens.new(bs, max_len + 1).fill_(self.pad_id)
        tgt_out = vq_tokens.new(bs, max_len + 1).fill_(self.pad_id)

        for i in range(bs):
            cur_len = skel_len[i].item()
            cur_point = vq_tokens[i, :cur_len]
            tgt_inp[i, 0] = self.bos_id
            tgt_inp[i, 1:1+cur_len] = cur_point
            tgt_out[i, :cur_len] = cur_point
            tgt_out[i, cur_len] = self.eos_id

        print("tgt_inp: ", tgt_inp[:2, :20])
        print("tgt_out: ", tgt_out[:2, :20])

        tgt_mask = self._get_mask(skel_len+1, max_len+1, vq_tokens.device).unsqueeze_(1).unsqueeze_(2)
        tgt_emb_inp = self.point_embedding(tgt_inp)
        tgt_emb_inp = self.pos_emb(tgt_emb_inp)
        out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask)
        out_logits = self.out_layer(out_feat) 

        out_logits = einops.rearrange(out_logits, "b t v -> (b t) v")
        tgt_out = tgt_out.view(-1)
        ce_loss = F.cross_entropy(out_logits, tgt_out, ignore_index=self.pad_id, reduction="none")
        ce_loss = ce_loss.sum() / tgt_mask.sum()

        return ce_loss


    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        self.vqvae.eval()
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        bs, max_len, v = points.size()
        
        points_mask = self._get_mask(skel_len*3, max_len*3, points.device)
        # vqvae encoder
        with torch.no_grad():
            vq_tokens, points_feat, commitment_loss = self.vqvae.vqvae_encode(points, points_mask) # [bs, t], [bs, h, t]

        ce_loss = self.compute_seq2seq_ce(gloss_id, vq_tokens, skel_len)

        self.log('{}_ce_loss'.format(mode), ce_loss.detach(), prog_bar=True)

        # total loss
        loss = ce_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        return loss


    def training_step(self, batch):
        loss = self.forward(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.vqvae.eval()
        if batch_idx < 10:
            gloss_id = batch["gloss_id"]   # [bs, src_len]
            gloss_len = batch["gloss_len"] # list(src_len)
            skel_len = batch["skel_len"]   # list(skel_len)
            ori_points = batch["skel_3d"]

            bs = gloss_id.size(0)

            pred_points, pred_len = self.generate(batch, batch_idx)
            _, pred_logits = self.back_translate_model2(pred_points, pred_len, gloss_id, gloss_len, "test")
            pred_logits = F.softmax(pred_logits, dim=-1) # [bs, t/4, vocab_size]
            pred_len = pred_len // 4
            pred_seq, _, _, out_seq_len = self.back_translate_model2.decoder.decode(pred_logits, pred_len)
            
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            for i, length in enumerate(gloss_len):
                ref = gloss_id[i][:length].tolist()[:-1]
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())][:-1]
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1
            test_res = dict(wer=err_delsubins, correct=correct, count=count)

            dtw_scores = []
            for i in range(bs):
                dec_point = pred_points[i, :pred_len[i].item(), :].cpu().numpy()
                ori_point = ori_points[i, :skel_len[i].item(), :].cpu().numpy()
                
                euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
                d, _, acc_cost_matrix, _ = dtw(dec_point, ori_point, dist=euclidean_norm)

                # Normalise the dtw cost by sequence length
                dtw_scores.append(d/acc_cost_matrix.shape[0])

            return test_res, dtw_scores


    def validation_epoch_end(self, outputs) -> None:
        test_err, test_correct, test_count = np.zeros([4]), 0, 0
        dtw_scores = []
        for test_out, dtw_score in outputs:
            test_err += test_out["wer"]
            test_correct += test_out["correct"]
            test_count += test_out["count"]
            dtw_scores.extend(dtw_score)

        self.log('{}/acc'.format("test"), test_correct / test_count, prog_bar=True)
        self.log('{}_wer'.format("test"), test_err[0] / test_count, prog_bar=True)
        self.log('{}_dtw_1'.format("test"), sum(dtw_scores) / len(dtw_scores), prog_bar=True)


    @torch.no_grad()
    def generate(self, batch, batch_idx, total_seq_len=300, temperature = 1., default_batch_size = 1, ):
        gloss_id = batch["gloss_id"]
        bs = gloss_id.size(0)

        src_feat = self.gloss_embedding(gloss_id) # [bs, src_len, emb_dim]
        src_feat = self.pos_emb(src_feat)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        res = gloss_id.new(bs, 1).fill_(self.bos_id) # [bs, 1]

        tgt_mask = None
        end_tag = gloss_id.new(bs).fill_(0).bool()
        pred_len = gloss_id.new(bs).fill_(1)

        for step in range(total_seq_len):
            res_emb = self.point_embedding(res)
            res_emb = res_emb + self.pos_emb(res_emb)

            max_pred_len = max(pred_len)
            pred_mask = self._get_mask(pred_len, max_pred_len, res.device)
            
            out_feat = self.transformer(src_feat, src_mask, res_emb, tgt_mask)
            logits = self.out_layer(out_feat)[:, -1, :] # [bs, vocab_size]
            if step % 3 != 0:
                logits[:, -3:] = float("-inf")
            sampled_idx = torch.argmax(logits, dim=-1) # [bs]
            eos_tag = (sampled_idx == self.eos_id)
            end_tag = eos_tag | end_tag
            if end_tag.all():
                break
            pred_len =  pred_len + (1-end_tag.long())
            res = torch.cat([res, sampled_idx.unsqueeze(1)], dim=-1)
        res = res[:, 1:]
        pred_len = pred_len - 1
        n_codes, emb_dim = self.vqvae.codebook.embeddings.size()
        embedding = torch.cat([self.vqvae.codebook.embeddings, torch.zeros(3, emb_dim).to(res.device)])
        pred_emb = F.embedding(res, embedding) # [bs, max_len, emb_dim]
        pred_feat = einops.rearrange(pred_emb, "b t h -> b h t")

        max_pred_len = max(pred_len)
        pred_mask = self._get_mask(pred_len, max_pred_len, res.device)

        dec_pose, dec_rhand, dec_lhand = self.vqvae.vqvae_decode(pred_feat, pred_mask.unsqueeze_(1).unsqueeze_(2))

        pred_points = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1)
        pred_points = einops.rearrange(pred_points, "(b t) v -> b t v", b=bs) # [b max_len/3 v]

        # ori_points = batch["skel_3d"]
        # if batch_idx < 2:
        #     for i in range(bs):
        #         pred_show_img = []
        #         ori_show_img = []
        #         pred_cur_points = pred_points[i, :tgt_len[i].item()].detach().cpu().numpy() # [cur_len, 150]
        #         ori_cur_points = ori_points[i, :tgt_len[i].item()].detach().cpu().numpy() # [cur_len, 150]
        #         for j in range(pred_cur_points.shape[0]):
        #             frame_joints = pred_cur_points[j]
        #             frame = np.ones((256, 256, 3), np.uint8) * 255
        #             frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
        #             # Draw the frame given 2D joints
        #             im = draw_frame_2D(frame, frame_joints_2d)
        #             pred_show_img.append(im)
        #         for j in range(ori_cur_points.shape[0]):
        #             frame_joints = ori_cur_points[j]
        #             frame = np.ones((256, 256, 3), np.uint8) * 255
        #             frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
        #             # Draw the frame given 2D joints
        #             im = draw_frame_2D(frame, frame_joints_2d)
        #             ori_show_img.append(im)
        #         pred_show_img = np.concatenate(pred_show_img, axis=1) # [h, w, c]
        #         ori_show_img = np.concatenate(ori_show_img, axis=1) # [h, w, c]
        #         show_img = np.concatenate([pred_show_img, ori_show_img], axis=0)
        #         cv2.imwrite("/Dataset/everybody_sign_now_experiments/predictions/epoch={}_batch={}_idx={}.png".format(self.current_epoch, batch_idx, i), show_img)
        #         # show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
        #         # show_img = torchvision.utils.make_grid(show_img, )
        #         # self.logger.experiment.add_image("{}/{}_batch_{}_{}".format("test", "pred", batch_idx, i), show_img, self.global_step)
        return pred_points, pred_len // 3

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    @torch.no_grad()
    def generate_step_with_probs(self, tgt_inp, tgt_mask, src_feat, src_mask):
        tgt_emb_inp = self.point_embedding(tgt_inp)
        tgt_emb_inp = self.pos_emb(tgt_emb_inp)

        out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask)
        out_logits = self.out_layer(out_feat) # [bs, max_len, vocab_size]
        out_logits[:, :, -2:] = float("-inf")
        probs = F.softmax(out_logits, dim=-1)
        max_probs, idx = probs.max(dim=-1)
        return idx, max_probs

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
        parser.add_argument('--backmodel', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=512)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        parser.add_argument('--backmodel2', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file2', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')

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
        dec_out = self.decoder(tgt_feat, tgt_mask, enc_out, src_mask)
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