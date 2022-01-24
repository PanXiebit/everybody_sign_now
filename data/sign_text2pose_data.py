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
        self.text_dict = text_dict
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
        tokens = self._tokens[idx] # IntTensor
        sent = self._sents[idx]
        keypoint_path = self._key_json_files[idx]
        keypoint_files = sorted(os.listdir(keypoint_path))[::2]
        points = []
        for keypoint_file in keypoint_files:
            posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(keypoint_path, keypoint_file))
            all_keypoints = np.array(posepts + facepts + r_handpts + l_handpts, dtype=np.float32) # 25, 
            points.append(np.expand_dims(all_keypoints, axis=0))
        
        keypoints = np.concatenate(points, axis=0)
        
        pose_anchor = [1]
        pose, pose_no_mask = self._get_x_y_and_normalize(keypoints[:, :75], pose_anchor) # [t, 3v]
        
        face_anchor = [33]
        face, face_no_mask = self._get_x_y_and_normalize(keypoints[:, 75:75+210], face_anchor)

        hand_anchor = [0]
        rhand, rhand_no_mask = self._get_x_y_and_normalize(keypoints[:, 75+210:75+210+63], hand_anchor)
        lhand, lhand_no_mask = self._get_x_y_and_normalize(keypoints[:, 75+210+63:75+210+63+63], hand_anchor)

        return dict(tokens=tokens,
                    pose=pose, pose_no_mask=pose_no_mask,
                    face=face, face_no_mask=face_no_mask,
                    rhand=rhand, rhand_no_mask=rhand_no_mask, 
                    lhand=lhand, lhand_no_mask=lhand_no_mask)

    def _get_x_y_and_normalize(self, points_array, anchor_ids):
        points_array = np.expand_dims(points_array, axis=0) # [1, T, 3*V]
        x_points = points_array[:, :, ::3]  # [1, T, V]
        y_points = points_array[:, :, 1::3] # [1, T, V]
        probs = points_array[:, :, 2::3]    # [1, T, V]

        no_mask = (probs != 0).astype(np.float32) # [1, T, V]

        # print(probs[:, :2, :], no_mask[:, :2, :])
        no_mask_anchor = no_mask[:, :, anchor_ids] # [1, T, 1]

        x_anchor = x_points[:, :, anchor_ids] # [1, T, 1]
        y_anchor = y_points[:, :, anchor_ids] # [1, T, 1]

        # # print(x_points[:, :2, :], y_points[:, :2, :])

        if (x_anchor == 0).any() or (y_anchor == 0).any():
            # print(x_anchor, y_anchor)
            x_anchor = np.mean(x_anchor) * (1 - no_mask_anchor) + x_anchor
            y_anchor = np.mean(y_anchor) * (1 - no_mask_anchor) + y_anchor


        x_points = ((x_points - x_anchor) / 1280.) * no_mask # [-1, 1]
        y_points = ((y_points - y_anchor) / 720.) * no_mask

        # x_points = ((x_points) / 640. - 1.) * no_mask
        # y_points = ((y_points) / 360. - 1.) * no_mask

        points = np.concatenate([x_points, y_points], axis=0) # [2, T, V]
        return torch.FloatTensor(points), torch.IntTensor(no_mask)

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

    def collate_fn(self, batch):        
        pose = self.collate_points([x["pose"] for x in batch], pad_idx=0)
        pose_no_mask = self.collate_points([x["pose_no_mask"] for x in batch], pad_idx=0)
        
        face = self.collate_points([x["face"] for x in batch], pad_idx=0)
        face_no_mask = self.collate_points([x["face_no_mask"] for x in batch], pad_idx=0)
        
        rhand = self.collate_points([x["rhand"] for x in batch], pad_idx=0)
        rhand_no_mask = self.collate_points([x["rhand_no_mask"] for x in batch], pad_idx=0)
        
        lhand = self.collate_points([x["lhand"] for x in batch], pad_idx=0)
        lhand_no_mask = self.collate_points([x["lhand_no_mask"] for x in batch], pad_idx=0)
        
        points_len = torch.IntTensor([x["pose"].size(0) for x in batch])

        tokens = self.collate_tokens([x["tokens"] for x in batch], pad_idx=self.text_dict.pad())
        tokens_len = torch.IntTensor([x["tokens"].size(0) for x in batch])
            
        return dict(points_len=points_len, tokens=tokens, tokens_len=tokens_len,
                    pose=pose, pose_no_mask=pose_no_mask,
                    face=face, face_no_mask=face_no_mask,
                    rhand=rhand, rhand_no_mask=rhand_no_mask, 
                    lhand=lhand, lhand_no_mask=lhand_no_mask)


    def collate_tokens(self, values, pad_idx, left_pad=False):
        
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    def collate_points(self, values, pad_idx, left_pad=False):
        """values[0].shape = [c, t, v]
        """
        
        c, _, v = values[0].shape
        size = max(v.size(1) for v in values)
        print("size: ", size)

        res = values[0].new(len(values), c, size, v).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        
        for i, v in enumerate(values):
            # print(v.shape, res[i][:, size - v.size(1):, :].shape)
            copy_tensor(v, res[i][:, size - v.size(1):, :] if left_pad else res[i][:, :v.size(1), :])
        return res
        

class How2SignTextPoseData(pl.LightningDataModule):

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
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()

    
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
        batchSize=5
        num_workers=32
        text_path=text_path
        vocab_file = vocab_file
        
    opts= Option()

    dataloader = How2SignTextPoseData(opts).train_dataloader()
    # dataloader = PoseSentDataset(opts)

    for i, data in enumerate(dataloader):
        if i > 20: break
        print(data["tokens"].shape)
        print(data["tokens_len"])
        print(data["pose"].shape)
        print(data["face"].shape)
        print(data["rhand"].shape)
        print(data["lhand"].shape)
        print(data["points_len"])
        print("-"*10)
    exit()
