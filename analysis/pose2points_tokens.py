import os
import itertools
import numpy as np
from pyrsistent import b
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from models.pose_vqvae_vit_model import PoseVitVQVAE
from configs.train_options import TrainOptions
from data.sign_text2pose_data import How2SignTextPoseData
from tqdm import tqdm


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        # if opts.fp16:
        #     return tensor.cuda(non_blocking=False).half()
        return tensor.cuda(non_blocking=False)

    return apply_to_sample(_move_to_cuda, sample)

pl.seed_everything(1234)
parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
opt = TrainOptions(parser).parse()
print(opt)

opt.data_path = "/data/xp_data/slr/EverybodySignNow/Data/how2sign"
opt.text_path = "data/text2gloss/"
opt.vocab_file = "data/text2gloss/how2sign_vocab.txt"
opt.batchSize = 5

vqvae_path = "logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}/lightning_logs/version_0/checkpoints/epoch=16-step=50965.ckpt"
hparams_file = "logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}/lightning_logs/version_0/hparams.yaml"
vqvae =  PoseVitVQVAE.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
vqvae.eval()


print(opt.data_path)

data = How2SignTextPoseData(opt)
train_loader = data.train_dataloader()

token_num = 5
with torch.no_grad():
    with open("analysis/text2point_tokens.txt", "w") as f:
        for batch in tqdm(train_loader):
            batch = move_to_cuda(batch)
            pose = batch["pose"]
            points_len = batch["points_len"].long() * token_num

            bs, c, t, _ = pose.size()
            points_tokens, points_embedding, _ = vqvae.encode(batch) # [bs, t, self.token_num]
            
            # print("points_tokens: ", points_tokens[:, :, 0:5])
            # print("points_tokens: ", points_tokens[:, :, 0:5].contiguous().view(bs, -1))
            
            pose_tokens = points_tokens[:, :, 0:5].contiguous().view(bs, -1)
            face_tokens = points_tokens[:, :, 5:10].contiguous().view(bs, -1)
            rhand_tokens = points_tokens[:, :, 10:15].contiguous().view(bs, -1)
            lhand_tokens = points_tokens[:, :, 15:20].contiguous().view(bs, -1)
            # print("points_tokens: ", points_tokens.shape)
            
            word_tokens = batch["tokens"].long()
            word_len = batch["tokens_len"].long()
            # print("points_len: ", points_len)

            bsz, _ = word_tokens.size()
            for id in range(bsz):
                cur_word = word_tokens[id, :word_len[id]].cpu().numpy().tolist()
                cur_pose = pose_tokens[id, :points_len[id]].cpu().numpy().tolist()
                cur_face = face_tokens[id, :points_len[id]].cpu().numpy().tolist()
                cur_rhand = rhand_tokens[id, :points_len[id]].cpu().numpy().tolist()
                cur_lhand = lhand_tokens[id, :points_len[id]].cpu().numpy().tolist()
                cur_word = " ".join([str(w) for w in cur_word])
                cur_pose = " ".join([str(w) for w in cur_pose])
                cur_face = " ".join([str(w) for w in cur_face])
                cur_rhand = " ".join([str(w) for w in cur_rhand])
                cur_lhand = " ".join([str(w) for w in cur_lhand])
                f.write(cur_word + " ||| " + cur_pose + " ||| " + cur_face + " ||| " + cur_rhand + " ||| " + cur_lhand + "\n")



