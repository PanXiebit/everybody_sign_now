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

opt.data_path = "generate/nat/"
# opt.data_path = "Data/ProgressiveTransformersSLP/"
opt.vocab_file = "Data/ProgressiveTransformersSLP/src_vocab.txt"


text_dict = Dictionary()
text_dict = text_dict.load(opt.vocab_file)

ctc_model = Point2textModel(opt, text_dict)

saved_path = "pose2text_logs/lightning_logs/val_wer_0.574/checkpoints/epoch=17-step=1007-val_wer=0.5748953819274902.ckpt"
hparams_file = "pose2text_logs/lightning_logs/val_wer_0.574/hparams.yaml"
ctc_model =  ctc_model.load_from_checkpoint(saved_path, hparams_file=hparams_file).cuda()


ctc_model.eval()


data = PhoenixPoseData(opt)
train_loader = data.test_dataloader()


with torch.no_grad():
    val_err, val_correct, val_count = np.zeros([4]), 0, 0
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        batch = move_to_cuda(batch)
        bs, t, v = batch["skel_3d"].size()
        if t < 4: continue
        out = ctc_model.validation_step(batch, batch_idx)

        val_err += out["wer"]
        val_correct += out["correct"]
        val_count += out["count"]

    print('{}/acc'.format("val"), val_correct / val_count)
    print('{}/wer'.format("val"), val_err[0] / val_count)
    print('{}/sub'.format("val"), val_err[1] / val_count)
    print('{}/ins'.format("val"), val_err[2] / val_count)
    print('{}/del'.format("val"), val_err[3] / val_count)

