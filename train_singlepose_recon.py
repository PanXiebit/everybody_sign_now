
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from data.sign_pose2pose_data import How2SignPoseData, PoseDataset
from models.pose_vqvae_single_vit_model import PoseSingleVQVAE
from util.util import CheckpointEveryNSteps



def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = PoseSingleVQVAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    print(opt)

    data = How2SignPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()
    model = PoseSingleVQVAE(opt)
    # model = model.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=1-step=2249.ckpt")
    
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor=None, filename='{epoch}-{step}', save_top_k=1))
    # callbacks.append(CheckpointEveryNSteps(6000, 2))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()