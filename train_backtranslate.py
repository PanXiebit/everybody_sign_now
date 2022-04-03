
from asyncio.log import logger
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models_phoneix.point2text_model import BackTranslateModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_phoneix.phoneix_text2pose_img_data_shift import PhoenixPoseData, PoseDataset
from util.util import CheckpointEveryNSteps
import os
from pytorch_lightning.loggers import NeptuneLogger
from data.vocabulary import Dictionary


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = BackTranslateModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    # print(opt)
    # print("opt.gpu_ids: ", opt.gpu_ids, type(opt.gpu_ids))
    # exit()

    data = PhoenixPoseData(opt)
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)

    model = BackTranslateModel(opt, text_dict)
    # model = model.load_from_checkpoint("/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/version_0/checkpoints/epoch=1-step=7095-val_wer=0.8766.ckpt", 
    #     hparams_file="/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/version_0/hparams.yaml")
    
    
    callbacks = []
    model_save_ccallback = ModelCheckpoint(monitor="val_wer", filename='{epoch}-{step}-{val_wer:.4f}', save_top_k=-1, mode="min")
    early_stop_callback = EarlyStopping(monitor="val_wer", min_delta=0.00, patience=20, verbose=False, mode="min")
    callbacks.append(model_save_ccallback)
    callbacks.append(early_stop_callback)

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)
    trainer.fit(model, data)
    # trainer.validate(model, dataloaders=data.test_dataloader())


if __name__ == "__main__":
    main()