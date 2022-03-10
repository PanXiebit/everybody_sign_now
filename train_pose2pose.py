
from asyncio.log import logger
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models.pose_vqvae_vit_model_pyramid import PoseVitVQVAE
from pytorch_lightning.callbacks import ModelCheckpoint
from data.sign_pose2pose_data import How2SignPoseData, PoseDataset
from util.util import CheckpointEveryNSteps
import os
from pytorch_lightning.loggers import NeptuneLogger


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = PoseVitVQVAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    # print(opt)
    # print("opt.gpu_ids: ", opt.gpu_ids, type(opt.gpu_ids))
    # exit()

    data = How2SignPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()
    model = PoseVitVQVAE(opt)
    # model = model.load_from_checkpoint("lightning_logs/seqlen_16_with_anchor/checkpoints/epoch=1-step=2249.ckpt", 
    #     hparams_file="lightning_logs/seqlen_16_with_anchor//hparams.yaml")
    
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