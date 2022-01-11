from genericpath import exists
import os
import os.path as osp
import math
import random
import pickle
from re import L
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

# POSE_MAX_X = 1280
# POSE_MAX_Y = 720
# POSE_MIN_X = -1280
# POSE_MIN_Y = -720


class PoseDataset(data.Dataset):
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

        tag = 'train' if train else 'val'
        csv_path = osp.join(opts.csv_path, 'how2sign_realigned_{}.csv'.format(tag))
        video_folder = os.path.join(data_path, tag, "videos")
        keypoint_folder = os.path.join(data_path, tag, "openpose_output/")

        data = pd.read_csv(csv_path, on_bad_lines='skip', delimiter="\t")
        
        debug = 5

        key_json_paths = []

        for i in tqdm(range(len(data))):
            if debug and i >= debug: break
            if "CTERDLghzFw_7-8-rgb_front" == data["SENTENCE_NAME"][i]: continue
            key_json_path = os.path.join(keypoint_folder, "json", data["SENTENCE_NAME"][i])
            try:
                assert os.path.exists(key_json_path), "{}".format(key_json_path)
            except:
                # print(data["SENTENCE_NAME"][i])
                continue
            key_json_paths.append(key_json_path)

        sequence_length = 16
        warnings.filterwarnings('ignore')

        keypoints_cache_file = osp.join(data_path, tag, f"metadata_lhand_{debug}_{sequence_length}.npy")

        if not os.path.exists(keypoints_cache_file):
            clips = []
            for keypoint_path in tqdm(key_json_paths):
                keypoint_files = sorted(os.listdir(keypoint_path))[::2]
                clip_data = []
                for keypoint_file in keypoint_files:
                    posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(keypoint_path, keypoint_file))
                    # print(len(posepts), len(facepts), len(r_handpts), len(l_handpts))
                    all_keypoints = np.array(posepts + facepts + r_handpts + l_handpts, dtype=np.float32) # 25, 
                    clip_data.append(np.expand_dims(all_keypoints, axis=0))
                    if len(clip_data) == sequence_length:
                        clips.append(np.expand_dims(np.concatenate(clip_data, axis=0), axis=0))
                        clip_data = []
            self._clips = np.concatenate(clips, axis=0)
            np.save(keypoints_cache_file, self._clips)
        else:
            self._clips = np.load(keypoints_cache_file)

    def __len__(self):
        return len(self._clips)


    def __getitem__(self, idx):
        keypoints = self._clips[idx]

        pose_anchor = [1]
        pose, pose_no_mask = self._get_x_y_and_normalize(keypoints[:, :75], pose_anchor)
        
        face_anchor = [33]
        face, face_no_mask = self._get_x_y_and_normalize(keypoints[:, 75:75+210], face_anchor)

        hand_anchor = [0]
        rhand, rhand_no_mask = self._get_x_y_and_normalize(keypoints[:, 75+210:75+210+63], hand_anchor)
        lhand, lhand_no_mask = self._get_x_y_and_normalize(keypoints[:, 75+210+63:75+210+63+63], hand_anchor)
        return dict(pose=pose, pose_no_mask=pose_no_mask,
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

        # print(x_points[:, :2, :], y_points[:, :2, :])

        if (x_anchor == 0).any() or (y_anchor == 0).any():
            # print(x_anchor, y_anchor)
            x_anchor = np.mean(x_anchor) * (1 - no_mask_anchor) + x_anchor
            y_anchor = np.mean(y_anchor) * (1 - no_mask_anchor) + y_anchor


        x_points = ((x_points - x_anchor) / 1280.) * no_mask
        y_points = ((y_points - y_anchor) / 720.) * no_mask


        points = np.concatenate([x_points, y_points], axis=0) # [2, T, V]
        return torch.FloatTensor(points), no_mask


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

class How2SignPoseData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        Dataset = PoseDataset
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


if __name__ == "__main__":
    pass
    csv_path = "Data"
    data_path = "/Dataset/how2sign/video_and_keypoint"
    class Option:
        hand_generator = True
        resolution = 256
        csv_path=csv_path
        data_path=data_path
        batchSize=1
        num_workers=32
    opts= Option()

    # dataloader = How2SignImagePairData(opts).train_dataloader()
    dataloader = PoseDataset(opts)

    for i, data in enumerate(dataloader):
        if i > 20: break
        print("")
        # print("pose: ", data["pose"].shape)
        # print("face: ", data["face"].shape)
        # print("rhand: ", data["rhand"].shape)
        # print("lhand: ", data["lhand"].shape)
    # exit()
