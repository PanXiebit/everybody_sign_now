
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from util.util import CheckpointEveryNSteps
from data.sign_video import How2SignImagePairData
from models.vq_cvae import VQCVAE
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = VQCVAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    print(opt)

    data = How2SignImagePairData(opt)
    data.train_dataloader()
    data.test_dataloader()

    model = VQCVAE(opt)
    # model = model.load_from_checkpoint("lightning_logs/version_0/checkpoints/Checkpoint_epoch=0_global_step=9000.ckpt")

    
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor=None, filename='{epoch}-{step}', save_top_k=1))
    callbacks.append(CheckpointEveryNSteps(1000, 2))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()