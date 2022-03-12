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
        data_path = opts.data_path
        text_path = opts.text_path
        self.max_frames_num = opts.max_frames_num

        text_dict = Dictionary()
        

        tag = 'train' if train else 'val'
        csv_path = osp.join(opts.csv_path, 'how2sign_realigned_{}.csv'.format(tag))
        text_path = osp.join(opts.text_path, 'how2sign.{}.norm.tok.en'.format(tag))
        # if tag == "train":
        #     text_dict._save_vocab_file(text_path, vocab_file)

        self.text_dict = text_dict
        text_dict = text_dict.load(opts.vocab_file)

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
            sent = sentences[i].lower().strip() # TODO, lower
            ids = text_dict.encode_line(sent)
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
        keypoint_files = sorted(os.listdir(keypoint_path))
        if len(keypoint_files) >= 8:
            keypoint_files = keypoint_files[::4] # TODO, sample rate
        points = []
        for keypoint_file in keypoint_files:
            posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(keypoint_path, keypoint_file))
            all_keypoints = np.array(posepts + facepts + r_handpts + l_handpts, dtype=np.float32) # 25, 
            if len(all_keypoints) == 411:
                points.append(np.expand_dims(all_keypoints, axis=0))
            else:
                print("error: ", os.path.join(keypoint_path, keypoint_file))
        keypoints = np.concatenate(points, axis=0)

        
        if len(points) % 4 != 0 or len(points) > self.max_frames_num:
            if len(points) < 4:
                # print("keypoints: ", keypoints.shape, keypoint_path)
                sample_size = 4
                samples = np.random.choice(list(range(len(points))), sample_size, replace=True)
                samples = sorted(samples.tolist())
                keypoints = keypoints[sorted(samples)]
                # print("keypoints: ", keypoints.shape, keypoint_path)
            else:
                sample_size = (len(points) // 4) * 4
                samples = random.sample(list(range(len(points))), min(sample_size, self.max_frames_num))
                keypoints = keypoints[sorted(samples)]
        
        
        pose = keypoints[:, :75]
        rhand = keypoints[:, 75+210:75+210+63]
        lhand = keypoints[:, 75+210+63:75+210+63+63]

        x_pose = pose[:, ::3]  # [T, V]
        y_pose = pose[:, 1::3] # [T, V]
        # print("x_pose : ", x_pose[:, 0])
        # print("y_pose : ", y_pose[:, 0])


        # print("x_anchor : ", x_pose[:, 1])
        # print("y_anchor : ", y_pose[:, 1])
        
        x_rhand = rhand[:, ::3]  # [T, V]
        y_rhand = rhand[:, 1::3] # [T, V]
        x_r_shift = x_pose[:, 4:5] - x_rhand[:, 0:1] # pose 4 is right
        y_r_shift = y_pose[:, 4:5] - y_rhand[:, 0:1] # [T, 1]
        x_rhand += x_r_shift
        y_rhand += y_r_shift

        
        x_lhand = lhand[:, ::3]  # [T, V]
        y_lhand = lhand[:, 1::3] # [T, V]
        x_l_shift = x_pose[:, 7:8] - x_lhand[:, 0:1] # pose 7 is right
        y_l_shift = y_pose[:, 7:8] - y_lhand[:, 0:1]
        x_lhand += x_l_shift
        y_lhand += y_l_shift

        x_anchor = x_pose[:, 1:2] # [T, 1], point 1 is in the center of the image.
        y_anchor = y_pose[:, 1:2]
        
        pose, pose_no_mask = self._get_x_y_and_normalize(x_pose, y_pose, x_anchor, y_anchor) # [t, 3v]
        rhand, rhand_no_mask = self._get_x_y_and_normalize(x_rhand, y_rhand, x_anchor, y_anchor)
        lhand, lhand_no_mask = self._get_x_y_and_normalize(x_lhand, y_lhand, x_anchor, y_anchor)

        return dict(tokens=tokens, tokens_len=len(tokens), sent=sent,
                    pose=pose, pose_no_mask=pose_no_mask,
                    rhand=rhand, rhand_no_mask=rhand_no_mask, 
                    lhand=lhand, lhand_no_mask=lhand_no_mask)


    def _get_x_y_and_normalize(self, x_points, y_points, x_anchor, y_anchor):

        no_mask = (x_points != 0).astype(np.float32) # [T, V]

        # if (x_anchor == 0).any() or (y_anchor == 0).any():
        #     # print(x_anchor, y_anchor)
        #     x_anchor = np.mean(x_anchor) * (1 - no_mask_anchor) + x_anchor
        #     y_anchor = np.mean(y_anchor) * (1 - no_mask_anchor) + y_anchor


        x_points = ((x_points - x_anchor) / 1280.) * no_mask # [-1, 1]
        y_points = ((y_points - y_anchor) / 720.) * no_mask

        x_points = np.expand_dims(x_points, axis=0)
        y_points = np.expand_dims(y_points, axis=0)

        points = np.concatenate([x_points, y_points], axis=0) # [2, T, V]
        return torch.FloatTensor(points), torch.IntTensor(np.expand_dims(no_mask, axis=0))


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
        
        rhand = self.collate_points([x["rhand"] for x in batch], pad_idx=0)
        rhand_no_mask = self.collate_points([x["rhand_no_mask"] for x in batch], pad_idx=0)
        
        lhand = self.collate_points([x["lhand"] for x in batch], pad_idx=0)
        lhand_no_mask = self.collate_points([x["lhand_no_mask"] for x in batch], pad_idx=0)
        
        points_len = torch.IntTensor([x["pose"].size(1) for x in batch])

        tokens = self.collate_tokens([x["tokens"] for x in batch], pad_idx=self.text_dict.pad())
        tokens_len = torch.IntTensor([x["tokens"].size(0) for x in batch])
            
        return dict(points_len=points_len, tokens=tokens, tokens_len=tokens_len,
                    pose=pose, pose_no_mask=pose_no_mask,
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
        size = max([v.size(1) for v in values])
        if size % 4 != 0: 
            size = math.ceil(size/4)*4  # TODO?
        res = values[0].new(len(values), c, size, v).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        
        for i, v in enumerate(values):
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
    data_path = "Data/how2sign"
    text_path = "data/text2gloss/"
    vocab_file = "data/text2gloss/how2sign_vocab.txt"
    class Option:
        hand_generator = True
        csv_path=csv_path
        data_path=data_path
        batchSize=5
        num_workers=32
        text_path=text_path
        vocab_file = vocab_file
        max_frames_num = 400
        
    opts= Option()

    
    # dataloader = PoseSentDataset(opts)
    # for i, data in enumerate(dataloader):
    #     if i > 5: break
    #     print(data["tokens"])
    #     # print(data["sent"])
    #     print(data["pose"].shape, data["pose_no_mask"].shape, data["pose"][:, :3, 7], data["pose"][:, :3, 4])
    #     print(data["rhand"].shape, data["rhand"][:, :3, 0])
    #     print(data["lhand"].shape, data["lhand"][:, :3, 0])
    #     print(data["tokens_len"])
    #     # print(data["points_len"])
    #     print("-"*10)
    # exit()

    dataloader = How2SignTextPoseData(opts).train_dataloader()
    for i, data in enumerate(dataloader):
        if i > 5: break
        print(data["tokens"])
        print(data["pose"].shape, data["pose"][0, :, :3, 7], data["pose"][0, :, :3, 4])
        print(data["rhand"].shape, data["rhand"][0, :, :3, 0])
        print(data["lhand"].shape, data["lhand"][0, :, :3, 0])
        print(data["tokens_len"])
        print(data["points_len"])
        print(data["pose_no_mask"][:, 0, :, 0].sum(-1)) # [bs, c, t, v]
        print(data["rhand_no_mask"][:, 0, :, 0].sum(-1)) # [bs, c, t, v]
        print(data["lhand_no_mask"][:, 0, :, 0].sum(-1)) # [bs, c, t, v]
        print("-"*10)
    exit()