
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse

from models_phoneix.point2text_model_vqvae_tr_nat_stage2_emb_seperate2 import Point2textModelStage2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_phoneix.phoneix_text2pose_data_shift import PhoenixPoseData
from util.util import CheckpointEveryNSteps
from data.vocabulary import Dictionary
import os



def main():
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = Point2textModelStage2.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    opt = TrainOptions(parser).parse()
    print(opt)

    # data = How2SignTextPoseData(opt)
    data = PhoenixPoseData(opt)
    data.train_dataloader()
    data.test_dataloader()

    text_dict = Dictionary()
    text_dict = text_dict.load(opt.vocab_file)
    model = Point2textModelStage2(opt, text_dict)
    if os.path.exists(opt.resume_ckpt):
        print("=== Load from {}!".format(opt.resume_ckpt))
        model = model.load_from_checkpoint(opt.resume_ckpt)
    else:
        print("=== {} is not existed!".format(opt.resume_ckpt))
    
    callbacks = []
    model_save_ccallback = ModelCheckpoint(monitor="test_wer", filename='{epoch}-{step}-{val_ce_loss:.4f}-{val_wer:4f}', save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="test_wer", min_delta=0.00, patience=100, verbose=False, mode="min")
    callbacks.append(model_save_ccallback)
    callbacks.append(early_stop_callback)

    kwargs = dict()
    if opt.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=opt.gpus)
    trainer = pl.Trainer.from_argparse_args(opt, callbacks=callbacks, 
                                            max_steps=200000000, **kwargs)

    trainer.validate(model, dataloaders=data.val_dataloader())
    # trainer.fit(model, data)


if __name__ == "__main__":
    main()