
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models_phoneix.text2pose_model_ctc_pretrain_nat_freeze_emb import Text2PoseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from data_phoneix.phoneix_text2pose_data_shift import PhoenixPoseData
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

    # data = How2SignTextPoseData(opt)
    data = PhoenixPoseData(opt)
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
    callbacks.append(ModelCheckpoint(monitor="val/mask_ce_loss", filename='{epoch}-{step}', save_top_k=-1))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()