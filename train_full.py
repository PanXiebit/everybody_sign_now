
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models.pix2pixHD_model import Pix2PixHDModel


def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    opt = TrainOptions(parser).parse()
    print(opt)

    model = Pix2PixHDModel(opt)


if __name__ == "__main__":
    main()