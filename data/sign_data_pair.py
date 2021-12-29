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



class ImagePairDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_path, video_folder, keypoint_folder, train=True, resolution=64):
        """
        Args:
            data_folder: "Data/"
            keypoint_folder: 
            video_folder: /Dataset/how2sign/video_and_keypoint/train/videos
        """
        super().__init__()
        self.train = train
        self.resolution = resolution

        tag = 'train' if train else 'val'
        data_path = osp.join(data_path, 'how2sign_realigned_{}.csv'.format(tag))
        video_folder = video_folder.replace("tag", tag)
        keypoint_folder = keypoint_folder.replace("tag", tag)

        data = pd.read_csv(data_path, on_bad_lines='skip', delimiter="\t")
        
        rgb_video_files = []
        kyp_video_files = []
        kyp_json_files = []
        for i in tqdm(range(len(data))):
            video_path = os.path.join(video_folder, data["SENTENCE_NAME"][i] + ".mp4")
            key_video_path = os.path.join(keypoint_folder, "video", data["SENTENCE_NAME"][i] + ".mp4")
            key_json_path = os.path.join(keypoint_folder, "json", data["SENTENCE_NAME"][i])
            try:
                assert os.path.exists(video_path), "{}".format(video_path)
                assert os.path.exists(key_video_path), "{}".format(key_video_path)
                assert os.path.exists(key_json_path), "{}".format(key_json_path)
            except:
                print(data["SENTENCE_NAME"][i])
                continue
            rgb_video_files.append(video_path)
            kyp_video_files.append(key_video_path)
            kyp_json_files.append(key_json_path)
        print("{} video number is: {}/{}".format(tag, len(rgb_video_files), len(data)))
        sequence_length = 1
        warnings.filterwarnings('ignore')
        rgb_cache_file = osp.join(video_folder, f"metadata_{sequence_length}.pkl")
        
        if not osp.exists(rgb_cache_file):
            rgb_clips = VideoClips(rgb_video_files, sequence_length, frames_between_clips=1, num_workers=4)
            pickle.dump(rgb_clips.metadata, open(rgb_cache_file, 'wb'))
        else:
            metadata = pickle.load(open(rgb_cache_file, 'rb'))
            rgb_clips = VideoClips(rgb_video_files, sequence_length, _precomputed_metadata=metadata)
        self.rgb_vid_clips = rgb_clips

        warnings.filterwarnings('ignore')
        key_cache_file = osp.join(keypoint_folder, f"metadata_{sequence_length}.pkl")
        
        if not osp.exists(key_cache_file):
            key_clips = VideoClips(rgb_video_files, sequence_length, frames_between_clips=1, num_workers=4)
            pickle.dump(key_clips.metadata, open(key_cache_file, 'wb'))
        else:
            metadata = pickle.load(open(key_cache_file, 'rb'))
            key_clips = VideoClips(rgb_video_files, sequence_length, _precomputed_metadata=metadata)
        self.key_vid_clips = key_clips

        print(len(self.rgb_vid_clips), len(self.key_vid_clips))
        exit()


    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        resolution = self.resolution
        video_path = self._files[idx]
        class_name = get_parent_dir(video_path)
        label = self.class_to_label[class_name]
        video = self.read_from_video_path(video_path)
        return dict(video=preprocess(video, resolution), label=label, path=video_path)

    def read_from_video_path(self, video_path):
        vid, start_f, end_f = [int(num) for num in os.path.basename(video_path).split(".")[0].split("_")]
        vidcap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # TODO, start should be 0. start -> 0
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i, offset in enumerate(range(total_frames)):
            success, img = vidcap.read()
            if success:
                pass
            else:
                continue
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        rgb_frames = frames[0:end_f - start_f + 1]
        # random_start sample

        random_start = True
        # random_step = True
        if random_start:
            if len(rgb_frames) >= self.sequence_length:
                select_from = range(0, len(rgb_frames) - self.sequence_length + 1)
                sample_start = random.choice(select_from)
                ids = list(range(sample_start, sample_start + self.sequence_length))
            else:
                ids = list(range(len(rgb_frames)))
                ids = ids + [ids[-1]] * (self.sequence_length - len(ids))
        # elif random_step:
        #     if len(rgb_frames) >= self.sequence_length:
        #         step = math.floor(len(rgb_frames) /self.sequence_length)
        #         ids = list(range(0, len(rgb_frames), step))
        #         ids.sort()
        #     else:
        #         ids = list(range(len(rgb_frames)))
        #         ids = ids + np.random.choice(ids, self.sequence_length - len(ids)).tolist()
        #         ids.sort()
        else:
            if len(rgb_frames) >= self.sequence_length:
                ids = random.sample(range(len(rgb_frames)), self.sequence_length)
                ids.sort()
            else:
                ids = list(range(len(rgb_frames)))
                ids = ids + np.random.choice(ids, self.sequence_length - len(ids)).tolist()
                ids.sort()
        rgb_frames = [rgb_frames[id] for id in ids]
        rgb_frames = torch.FloatTensor(rgb_frames)
        return rgb_frames


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))

def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    # normalize the images between -1 and 1
    video = video.permute(0, 3, 1, 2).float() / 255. # THWC -> TCHW
    # video = (video.permute(0, 3, 1, 2).float() / 127.5 - 1) # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5
    return video # [bs, c, t, h, w]


class How2SignImagePairData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, train):
        Dataset = ImagePairDataset
        dataset = Dataset(self.args.data_path, self.args.sequence_length,
                          train=train, resolution=self.args.resolution)
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
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    data_path = "Data"
    video_folder = "/Dataset/how2sign/video_and_keypoint/train/videos"
    keypoint_folder="/Dataset/how2sign/video_and_keypoint/train/openpose_output"
    dataset = ImagePairDataset(data_path, video_folder, keypoint_folder,)