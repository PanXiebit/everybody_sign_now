
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models.pix2pixHD_model import Pix2PixHDModel
from pytorch_lightning.callbacks import ModelCheckpoint
from data.sign_data_pair import How2SignImagePairData

def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    opt = TrainOptions(parser).parse()
    print(opt)

    data = How2SignImagePairData(opt)
    data = data.train_dataloader()
    model = Pix2PixHDModel(opt)
    
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min',
        filename='{epoch}-{step}', save_top_k=1))

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.fit(model, data)


if __name__ == "__main__":
    main()