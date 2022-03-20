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
from models_phoneix.pose_vqvae_vit_model_spl_seperate import PoseVitVQVAE
from configs.train_options import TrainOptions
from data.phoneix_text2pose_data_shift import PhoenixPoseData
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

opt.data_path = "Data/ProgressiveTransformersSLP"
opt.vocab_file = "Data/ProgressiveTransformersSLP/src_vocab.txt"
opt.batchSize = 5

vqvae_path = "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/checkpoints/epoch=87-step=35551.ckpt"
hparams_file = "logs/phoneix_spl_seperate_SeqLen_1/lightning_logs/version_3/hparams.yaml"
vqvae =  PoseVitVQVAE.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
vqvae.eval()


print(opt.data_path)

data = PhoenixPoseData(opt)
mode = "train"
if mode == "val":
    train_loader = data.val_dataloader()
else:
    train_loader = data.train_dataloader()

token_num = 5
with torch.no_grad():
    with open("analysis/spl_phoneix/text2point_tokens_spl_{}.word".format(mode), "w") as f1, \
        open("analysis/spl_phoneix/text2point_tokens_spl_{}.point".format(mode), "w") as f2:
        for batch in tqdm(train_loader):
            batch = move_to_cuda(batch)
            points = batch["skel_3d"]  # [bs, 150]
            bs, t, v = points.size()
            points = points.view(bs*t, v)
            pose = points[:, :24]
            rhand = points[:, 24:24+63]
            lhand = points[:, 87:150]

            points_tokens, _, _ = vqvae.encode(pose, rhand, lhand) # [bs, 1]
            points_tokens = einops.rearrange(points_tokens, "(b t) n -> b (t n)", b=bs)
            
            points_len = batch["skel_len"]

            word_tokens = batch["gloss_id"].long() * 3
            word_len = batch["gloss_len"]

            bsz, _ = word_tokens.size()
            for id in range(bsz):
                cur_word = word_tokens[id, :word_len[id]].cpu().numpy().tolist()
                cur_point = points_tokens[id, :points_len[id]].cpu().numpy().tolist()

                cur_word = " ".join([str(w) for w in cur_word])
                cur_point = " ".join([str(w) for w in cur_point])
                f1.write(cur_word + "\n")
                f2.write(cur_point + "\n")



