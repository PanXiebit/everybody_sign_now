
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models.pix2pixHD_model import Pix2PixHDModel
from pytorch_lightning.callbacks import ModelCheckpoint
from data.sign_pose2rgb import How2SignImagePairData
from util.util import CheckpointEveryNSteps



def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    opt = TrainOptions(parser).parse()
    print(opt)

    data = How2SignImagePairData(opt)
    data.train_dataloader()
    data.test_dataloader()
    model = Pix2PixHDModel(opt)
    model = model.load_from_checkpoint("lightning_logs/version_1/checkpoints/Checkpoint_epoch=0_global_step=72000.ckpt")
    
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor=None, filename='{epoch}-{step}', save_top_k=1))
    callbacks.append(CheckpointEveryNSteps(6000, 2))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()