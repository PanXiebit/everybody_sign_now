from genericpath import exists
import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import h5py
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import cv2
import pandas as pd
from tqdm import tqdm
from torchvision.datasets.video_utils import VideoClips
import json
from PIL import Image
from .data_prep.renderopenpose import makebox128, fix_scale_image, fix_scale_coords, scale_resize
import torchvision.transforms as transforms
from collections import defaultdict
from .vocabulary import Dictionary


class PoseSentDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, opts, train=True):
        """
        Args:
            csv_path: Data/
            data_folder: "/Dataset/how2sign/video_and_keypoint/"
            keypoint_folder: 
            video_folder: /Dataset/how2sign/video_and_keypoint/train/videos
        """
        super().__init__()
        self.train = train
        # self.hand_generator = opts.hand_generator
        data_path = opts.data_path
        text_path = opts.text_path

        text_dict = Dictionary()
        
        # vocabulary._save_vocab_file(tokenized_sent_path, vocab_file)
        text_dict = text_dict.load(opts.vocab_file)

        tag = 'train' if train else 'val'
        csv_path = osp.join(opts.csv_path, 'how2sign_realigned_{}.csv'.format(tag))
        text_path = osp.join(opts.text_path, 'how2sign.{}.norm.tok.en'.format(tag))
        keypoint_folder = os.path.join(data_path, tag, "openpose_output/")

        data = pd.read_csv(csv_path, on_bad_lines='skip', delimiter="\t")
        with open(text_path, "r") as f:
            sentences = f.readlines()
        assert len(sentences) == len(data)

        debug = 0
        warnings.filterwarnings('ignore')

        key_json_files = []
        tokens = []
        sents = []

        for i in tqdm(range(len(data))):
            if debug and i >= debug: break
            # if "CTERDLghzFw_7-8-rgb_front" == data["SENTENCE_NAME"][i]: 
            #     continue
            key_json_path = os.path.join(keypoint_folder, "json", data["SENTENCE_NAME"][i])
            sent = sentences[i].strip()
            ids = text_dict.encode_line(sentences[i])
            try:
                assert os.path.exists(key_json_path), "{}".format(key_json_path)
            except:
                # print(data["SENTENCE_NAME"][i])
                continue

            key_json_files.append(key_json_path)
            tokens.append(ids)
            sents.append(sent)

        self._tokens = tokens
        self._key_json_files = key_json_files
        self._sents = sents

        assert len(tokens) == len(key_json_files)
        print("{} video number is: {}/{}".format(tag, len(tokens), len(data)))

        
    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        tokens = self._tokens[idx]
        sent = self._sents[idx]
        keypoint_path = self._key_json_files[idx]
        keypoint_files = sorted(os.listdir(keypoint_path))[::2]
        points = []
        for keypoint_file in keypoint_files:
            posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(keypoint_path, keypoint_file))
            all_keypoints = np.array(posepts + facepts + r_handpts + l_handpts, dtype=np.float32) # 25, 
            points.append(np.expand_dims(all_keypoints, axis=0))
        points = np.concatenate(points, axis=0)

        return dict(points=points, sent=sent, tokens=tokens)

    def normalize(self, vid):
        img = vid.float() / 127.5
        return img - 1.0

    def readkeypointsfile_json(self, myfile):
        f = open(myfile, 'r')
        json_dict = json.load(f)
        people = json_dict['people']
        posepts =[]
        facepts = []
        r_handpts = []
        l_handpts = []
        for p in people:
            posepts += p['pose_keypoints_2d']
            facepts += p['face_keypoints_2d']
            r_handpts += p['hand_right_keypoints_2d']
            l_handpts += p['hand_left_keypoints_2d']
        return posepts, facepts, r_handpts, l_handpts
        

class How2SignImagePairData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        Dataset = PoseSentDataset
        dataset = Dataset(self.args, train=train)
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batchSize,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=False
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()

    def collate_fn(self, batch):
        pass


if __name__ == "__main__":
    pass
    csv_path = "Data"
    data_path = "/Dataset/how2sign/video_and_keypoint"
    text_path = "data/text2gloss/"
    vocab_file = "data/text2gloss/how2sign_vocab.txt"
    class Option:
        hand_generator = True
        resolution = 256
        csv_path=csv_path
        data_path=data_path
        batchSize=1
        num_workers=32
        text_path=text_path
        vocab_file = vocab_file
        
    opts= Option()

    # dataloader = How2SignImagePairData(opts).train_dataloader()
    dataloader = PoseSentDataset(opts)

    for i, data in enumerate(dataloader):
        if i > 20: break
        print(data["points"].shape)
        print(data["sent"])
        print(data["tokens"].shape)
    exit()
