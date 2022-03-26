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
# from models_phoneix.text2pose_model_ctc_pretrain import Text2PoseModel
from models_phoneix.point2text_model_vqvae import Point2textModel
from configs.train_options import TrainOptions
from data_phoneix.phoneix_text2pose_data_shift import PhoenixPoseData
from tqdm import tqdm
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
opt.batchSize = 4

# vqvae_path = "text2pose_logs/phoneix_seperate/lightning_logs/saved_ver/checkpoints/epoch=39-step=35479.ckpt"
# vqvae_path = "text2pose_logs/phoneix_seperate/lightning_logs/with_ctc/checkpoints/epoch=279-step=248359.ckpt"
vqvae_path = "/Dataset/everybody_sign_now_experiments/pose2text_logs/lightning_logs/version_2/checkpoints/epoch=9-step=5919-val_wer=0.628754734992981.ckpt"
hparams_file = "/Dataset/everybody_sign_now_experiments/pose2text_logs/lightning_logs/version_2/hparams.yaml"
model = Point2textModel.load_from_checkpoint(vqvae_path, hparams_file=hparams_file).cuda()
model.eval()


text_dict = model.text_dict

data = PhoenixPoseData(opt)
mode = "train"
if mode == "dev":
    train_loader = data.test_dataloader()
else:
    train_loader = data.train_dataloader()

with torch.no_grad():
    with open("generate/nat/{}.tokens".format(mode), "w") as f1, open("generate/nat/{}.tokens.rm".format(mode), "w") as f2, open("generate/nat/{}.gloss".format(mode), "w") as f3:
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            # if batch_idx > 5: break
            batch = move_to_cuda(batch)
            gloss_id = batch["gloss_id"]
            gloss_len = batch["gloss_len"]
            skel_len = batch["skel_len"]
            _, _, vq_tokens = model(batch, "test") # [t, 150]

            for i in range(gloss_id.size(0)):
                
                pred_tokens = vq_tokens[i, :skel_len[i]].cpu().numpy().tolist() # [max_len * 3]
                rm_dup_tokens = [pred_tokens[0]]
                for tok in pred_tokens[1:]:
                    if tok != rm_dup_tokens[-1]:
                        rm_dup_tokens.append(tok)
                pred_tokens_line = " ".join([str(w) for w in pred_tokens])
                rm_dup_tokens_line = " ".join([str(w) for w in rm_dup_tokens])
                f1.write(pred_tokens_line + "\n")
                f2.write(rm_dup_tokens_line + "\n")

                word_tokens = gloss_id[i, :gloss_len[i]].cpu().numpy().tolist()[:-1]
                word_line = text_dict.deocde_list(word_tokens)
                f3.write(word_line + "\n")





            
                


 