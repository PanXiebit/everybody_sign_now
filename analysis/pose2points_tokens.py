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
from models.pose_vqvae_vit_model_spl_joint import PoseVitVQVAE
from configs.train_options import TrainOptions
from data.sign_text2pose_data_shift import How2SignTextPoseData
from tqdm import tqdm
import einops


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

opt.data_path = "Data/how2sign"
opt.text_path = "data/text2gloss/"
opt.vocab_file = "data/text2gloss/how2sign_vocab.txt"
opt.batchSize = 5

vqvae_path = "logs/spl_joint_SeqLen_1/lightning_logs/version_0/checkpoints/epoch=12-step=2053.ckpt"
hparams_file = "logs/spl_joint_SeqLen_1/lightning_logs/version_0/hparams.yaml"
vqvae =  PoseVitVQVAE.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
vqvae.eval()


print(opt.data_path)

data = How2SignTextPoseData(opt)
mode = "val"
if mode == "val":
    train_loader = data.val_dataloader()
else:
    train_loader = data.train_dataloader()

token_num = 5
with torch.no_grad():
    with open("analysis/spl_joint/text2point_tokens_spl_{}.word".format(mode), "w") as f1, \
        open("analysis/spl_joint/text2point_tokens_spl_{}.point".format(mode), "w") as f2:
        for batch in tqdm(train_loader):
            batch = move_to_cuda(batch)
            pose = batch["pose"][..., [1,0,2,3,4,5,6,7]] # [bs, c, t, v]
            rhand = batch["rhand"]
            lhand = batch["lhand"]
            bs, _, t, _ = pose.size()
            # print(pose.shape, rhand.shape, lhand.shape)
            vq_pose = einops.rearrange(pose, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_rhand = einops.rearrange(rhand, "b c t v -> (b t) c v").unsqueeze(-2)
            vq_lhand = einops.rearrange(lhand, "b c t v -> (b t) c v").unsqueeze(-2)
            # print(pose.shape, rhand.shape, lhand.shape)
            points_tokens, points_embedding, _ = vqvae.encode(vq_pose, vq_rhand, vq_lhand) # [bs*t, 3]
            # print("points_tokens: ", points_tokens.shape) 
            points_tokens = einops.rearrange(points_tokens, "(b t) n -> b (t n)", b=bs, n=1)
            points_len = batch["points_len"]

            word_tokens = batch["tokens"].long()
            word_len = batch["tokens_len"].long()
            
            bsz, _ = word_tokens.size()
            for id in range(bsz):
                cur_word = word_tokens[id, :word_len[id]].cpu().numpy().tolist()
                cur_point = points_tokens[id, :points_len[id]].cpu().numpy().tolist()

                cur_word = " ".join([str(w) for w in cur_word])
                cur_point = " ".join([str(w) for w in cur_point])
                f1.write(cur_word + "\n")
                f2.write(cur_point + "\n")



