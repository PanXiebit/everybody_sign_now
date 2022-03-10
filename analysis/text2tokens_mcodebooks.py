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
from models.text2pose_model_separate_slice_ar import Text2PoseModel
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
opt.batchSize = 2

vqvae_path = "text2pose_logs/separate_ar/lightning_logs/version_3/checkpoints/epoch=2-step=23285.ckpt"
hparams_file = "text2pose_logs/separate_ar/lightning_logs/version_3/hparams.yaml"
text2pose_model =  Text2PoseModel.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
text2pose_model.eval()


print(opt.data_path)

data = How2SignTextPoseData(opt)
train_loader = data.test_dataloader()

token_num = 20
with torch.no_grad():
    with open("analysis/text2tokens_mcodebooks_teacher_len4.txt", "w") as f:
        for batch in tqdm(train_loader):
            batch = move_to_cuda(batch)
            pose = batch["pose"]
            bs, c, t, _ = pose.size()
            points_token_pred, predict_len_pred = text2pose_model.inference_fast(batch) # [bs, t//4, self.token_num]
            
            points_tokens_rec, _, _ = text2pose_model.vqvae.encode(batch)# print("pose_tokens: ", pose_tokens.shape)

            
            points_len_rec = batch["points_len"].long()

            # print("length: ", points_len_rec, predict_len_pred)
            # exit()
            # pose_tokens = points_tokens[:, :, 0:5].contiguous().view(bs, -1)
            # face_tokens = points_tokens[:, :, 5:10].contiguous().view(bs, -1)
            # rhand_tokens = points_tokens[:, :, 10:15].contiguous().view(bs, -1)
            # lhand_tokens = points_tokens[:, :, 15:20].contiguous().view(bs, -1)
            
            word_tokens = batch["tokens"].long()
            word_len = batch["tokens_len"].long()
            

            bsz, _ = word_tokens.size()
            for id in range(bsz):
                cur_word = word_tokens[id, :word_len[id]].cpu().numpy().tolist()
                cur_word = " ".join([str(w) for w in cur_word])
                f.write(cur_word + "\n")
                for j in range(token_num):
                    cur_pose_rec = points_tokens_rec[id, :points_len_rec[id], j].cpu().numpy().tolist()
                    cur_pose_pred = points_token_pred[id, :predict_len_pred[id], j].cpu().numpy().tolist()
                    cur_pose_rec = " ".join([str(w) for w in cur_pose_rec])
                    cur_pose_pred = " ".join([str(w) for w in cur_pose_pred])
                    f.write(str(j)*10 + " === " + cur_pose_rec + "\n")
                    f.write(str(j)*10 + " === " + cur_pose_pred + "\n")



 