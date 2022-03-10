
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models.text2pose_nar_model import Text2PoseModel
from models.text2pose_model_separate_ar_vec2 import Text2PoseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from data.sign_text2pose_data import How2SignTextPoseData
from util.util import CheckpointEveryNSteps
from data.vocabulary import Dictionary
import os


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = Text2PoseModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    print(opt)

    data = How2SignTextPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)
    model = Text2PoseModel(opt, text_dict)
    if os.path.exists(opt.resume_ckpt):
        print("=== Load from {}!".format(opt.resume_ckpt))
        model = model.load_from_checkpoint(opt.resume_ckpt)
    else:
        print("=== {} is not existed!".format(opt.resume_ckpt))
    # print(model)
    # exit()
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val/loss", filename='{epoch}-{step}', save_top_k=1))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()