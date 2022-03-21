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
from models_phoneix.text2pose_model_ctc import Text2PoseModel
from configs.train_options import TrainOptions
from data_phoneix.phoneix_text2pose_data_shift import PhoenixPoseData
from tqdm import tqdm
from models_phoneix.point2text_model import Point2textModel
from data.vocabulary import Dictionary


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
opt.batchSize = 1

vqvae_path = "text2pose_logs/phoneix_seperate/lightning_logs/saved_ver/checkpoints/epoch=39-step=35479.ckpt"
hparams_file = "text2pose_logs/phoneix_seperate/lightning_logs/saved_ver/hparams.yaml"
text2pose_model =  Text2PoseModel.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
text2pose_model.eval()


text_dict = text2pose_model.text_dict

data = PhoenixPoseData(opt)
train_loader = data.test_dataloader()

with torch.no_grad():
    with open("generate/dev.skels", "w") as fs, open("generate/dev.gloss", "w") as fg:
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            # if batch_idx > 5: break
            batch = move_to_cuda(batch)
            res_tokens, pred_points = text2pose_model.generate(batch, batch_idx) # [t, 150]
            t, v = pred_points.shape
            pred_points = np.concatenate((pred_points, np.ones((t, 1))), axis=1)
            pred_points = pred_points.reshape(t*(v+1)).tolist()
            pred_points_line = " ".join([str(w) for w in pred_points])
            fs.write(pred_points_line + "\n")
            word_tokens = batch["gloss_id"].view(-1).cpu().numpy().tolist()[:-1]
            word_line = text_dict.deocde_list(word_tokens)
            fg.write(word_line + "\n")




            
                


 