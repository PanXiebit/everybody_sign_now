
from asyncio.log import logger
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models_phoneix.point2text_model import Point2textModel
from pytorch_lightning.callbacks import ModelCheckpoint
from data_phoneix.phoneix_text2pose_data_shift import PhoenixPoseData, PoseDataset
from util.util import CheckpointEveryNSteps
import os
from pytorch_lightning.loggers import NeptuneLogger
from data.vocabulary import Dictionary


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = Point2textModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    # print(opt)
    # print("opt.gpu_ids: ", opt.gpu_ids, type(opt.gpu_ids))
    # exit()

    data = PhoenixPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)

    model = Point2textModel(opt, text_dict)
    # model = model.load_from_checkpoint("lightning_logs/seqlen_16_with_anchor/checkpoints/epoch=1-step=2249.ckpt", 
    #     hparams_file="lightning_logs/seqlen_16_with_anchor//hparams.yaml")
    
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val/wer", filename='{epoch}-{step}', save_top_k=1, mode="min"))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()