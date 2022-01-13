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


class ImagePairDataset(data.Dataset):
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
        style_img_folder = os.path.join(data_path, tag, "style_images/")
        self.style_img_folder = style_img_folder

        data = pd.read_csv(csv_path, on_bad_lines='skip', delimiter="\t")
        
        sequence_length = 4
        frames_between_clips = 8
        debug = 0
        warnings.filterwarnings('ignore')
        rgb_cache_file = osp.join(data_path, tag, f"rgb_vid_metadata_{debug}_{sequence_length}_{frames_between_clips}.pkl")
        key_cache_file = osp.join(data_path, tag, f"kyp_vid_metadata_{debug}_{sequence_length}_{frames_between_clips}.pkl")
        key_rhand_cache_file = osp.join(data_path, tag, f"metadata_rhand_{debug}_{sequence_length}_{frames_between_clips}.txt")
        key_lhand_cache_file = osp.join(data_path, tag, f"metadata_lhand_{debug}_{sequence_length}_{frames_between_clips}.txt")

        rgb_video_files = []
        kyp_video_files = []
        key_json_files = []
        style_image_files = []

        for i in tqdm(range(len(data))):
            if debug and i >= debug: break
            if "CTERDLghzFw_7-8-rgb_front" == data["SENTENCE_NAME"][i]: continue
            video_path = os.path.join(video_folder, data["SENTENCE_NAME"][i] + ".mp4")
            key_video_path = os.path.join(keypoint_folder, "video", data["SENTENCE_NAME"][i] + ".mp4")
            key_json_path = os.path.join(keypoint_folder, "json", data["SENTENCE_NAME"][i])
            style_image_file = os.path.join(style_img_folder, data["SENTENCE_NAME"][i] + ".jpg")
            try:
                assert os.path.exists(video_path), "{}".format(video_path)
                assert os.path.exists(key_video_path), "{}".format(key_video_path)
                assert os.path.exists(key_json_path), "{}".format(key_json_path)
                assert os.path.exists(style_image_file), "{}".format(style_image_file)
            except:
                # print("missing: ", data["SENTENCE_NAME"][i])
                continue
            rgb_video_files.append(video_path)
            kyp_video_files.append(key_video_path)
            key_json_files.append(key_json_path)
            style_image_files.append(style_image_file)
        self.style_image_files = style_image_files

        assert len(rgb_video_files) == len(kyp_video_files) == len(key_json_files)
        print("{} video number is: {}/{}".format(tag, len(rgb_video_files), len(data)))
        
        if not osp.exists(rgb_cache_file):
            rgb_clips = VideoClips(rgb_video_files, sequence_length, frames_between_clips=frames_between_clips, num_workers=24)
            pickle.dump(rgb_clips.metadata, open(rgb_cache_file, 'wb'))
        else:
            metadata = pickle.load(open(rgb_cache_file, 'rb'))
            rgb_clips = VideoClips(rgb_video_files, sequence_length, frames_between_clips=frames_between_clips, _precomputed_metadata=metadata)
        self.rgb_vid_clips = rgb_clips

        warnings.filterwarnings('ignore')        
        if not osp.exists(key_cache_file):
            key_clips = VideoClips(kyp_video_files, sequence_length, frames_between_clips=frames_between_clips, num_workers=24)
            pickle.dump(key_clips.metadata, open(key_cache_file, 'wb'))
        else:
            metadata = pickle.load(open(key_cache_file, 'rb'))
            key_clips = VideoClips(kyp_video_files, sequence_length, frames_between_clips=frames_between_clips, _precomputed_metadata=metadata)
        self.key_vid_clips = key_clips

        print("rgb_videos, key_videos clip number: {}, {}".format(len(self.rgb_vid_clips), len(self.key_vid_clips)))

        if not osp.exists(key_rhand_cache_file) or not osp.exists(key_lhand_cache_file):
            with open(key_rhand_cache_file, "w") as rw, open(key_lhand_cache_file, "w") as lw:
                for k, key_json_path in tqdm(enumerate(key_json_files)):
                    json_files = sorted(os.listdir(key_json_path))[:-(sequence_length-1)][::frames_between_clips]
                    for json_file in json_files:
                        posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(key_json_path, json_file))
                        rw.write(" ".join([str(k) for k in r_handpts]) + "\n")
                        lw.write(" ".join([str(k) for k in l_handpts]) + "\n")
        self.kyp_rhands = open(key_rhand_cache_file, "r").readlines()
        self.kyp_lhands = open(key_lhand_cache_file, "r").readlines()
        print("rhand, lhand clip number: ", len(self.kyp_rhands), len(self.kyp_lhands))

    def __len__(self):
        return len(self.key_vid_clips)

    def __getitem__(self, idx):
        label, _, _, label_idx, key_vid_0 = self.key_vid_clips.get_clip(idx) # change the source code: add video_path as an output.
        label = self.normalize(label) # [t, c, y, x]

        rgb, _, _, rgb_idx, rgb_path_0 = self.rgb_vid_clips.get_clip(idx)
        rgb = self.normalize(rgb)

        style_img_path = os.path.join(self.style_img_folder, os.path.basename(key_vid_0)[:-4] + ".jpg")
        style_img = cv2.cvtColor(cv2.imread(style_img_path), cv2.COLOR_BGR2RGB) 
        style_img = torch.FloatTensor(style_img).permute(2, 0, 1)
        style_img = (style_img / 127.5) - 1.  # normalize

        # print("key_vid_0: ", os.path.basename(key_vid_0))
        # print("rgb_path_0: ", os.path.basename(rgb_path_0))
        # print("style_img: ", os.path.basename(style_img_path), style_img.shape)

        # assert label_idx == rgb_idx, (label_idx, rgb_idx)
        assert os.path.basename(key_vid_0) == os.path.basename(rgb_path_0), (os.path.basename(key_vid_0), os.path.basename(rgb_path_0))
        
        rhands_0 = list(map(float, self.kyp_rhands[idx].strip().split()))
        lhands_0 = list(map(float, self.kyp_lhands[idx].strip().split()))

        # crop image
        (label, rgb), style_img, (lhands_0, rhands_0) = self.crop_and_resize(
            [label, rgb], style_img, (720, 720), (256, 256), [lhands_0, rhands_0])

        shape = rgb.shape[-2:]
        lhand_coords = self._get_hands(shape, lhands_0, idx, "hand_left")
        rhand_coords = self._get_hands(shape, rhands_0, idx, "hand_right")

        return dict(label=label, rgb=rgb, style_img=style_img,
                    lhand_coords=lhand_coords,
                    rhand_coords=rhand_coords)

    def normalize(self, vid):
        img = vid.permute(3, 0, 1, 2).float() / 127.5  # c, t, h, w 
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

    def crop_and_resize(self, videos, style_img, cropped_shape, resize_shape, hands_points):
        # crop and resize
        t, c, y, x = videos[0].shape # [720, 1280]

        crop_y, crop_x = cropped_shape # [720, 720]

        shift_y, start_y, end_y = 0, 0, y
        shift_x, start_x, end_x = 0, 0, x
        if crop_y < y:
            shift_y = (y - crop_y) //2
            start_y, end_y = shift_y, shift_y + crop_y
        if crop_x < x:
            shift_x = (x - crop_x) // 2
            start_x, end_x = shift_x, shift_x + crop_x
        
        videos = [vid[:, :, start_y:end_y, start_x:end_x] for vid in videos]
        videos = [F.interpolate(vid, size=resize_shape, mode='bilinear', align_corners=False) for vid in videos]
        style_img = style_img[:, start_y:end_y, start_x:end_x]
        style_img = F.interpolate(style_img.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=False)[0]

        scale = resize_shape[0] / cropped_shape[0]
        hands_points = [fix_scale_coords(points, scale, (-shift_x, -shift_y)) for points in hands_points]
        return videos, style_img, hands_points


    def _get_hands(self, shape, lhand_points, idx, name):
        """
            len(hand_points) = 21 * 3
            img.shape: [3, 256, 256]
        """
        nodes = [0, 4, 8, 12, 16, 20]

        avex, avey = [], []
        for id in nodes:
            if lhand_points[id*3 + 2] > 0:
                avex.append(lhand_points[id*3]) # x
                avey.append(lhand_points[id*3+1]) # y

        if len(avex) == 0:
            avex, avey = 0., 0.
        else:
            avex = sum(avex) / len(avex)
            avey = sum(avey) / len(avey)
        

        boxbuffer = 80
        startx, starty = 0, 0
        endy, endx = shape # 720, 1280
        scalex, scaley = 1.0, 1.0
        minx = int((max(avex - boxbuffer, startx) - startx) * scalex)
        miny = int((max(avey - boxbuffer, starty) - starty) * scaley)
        maxx = int((min(avex + boxbuffer, endx) - startx) * scalex)
        maxy = int((min(avey + boxbuffer, endy) - starty) * scaley)

        miny, maxy, minx, maxx = makebox128(miny, maxy, minx, maxx, 64, 64, endy, endx)
        # hand = img[:, miny:maxy, minx:maxx]

        # im = Image.fromarray(hand.permute(1,2,0).contiguous().numpy())
        # im.save("Data/hand128x128/{}_{}.png".format(name, idx))
        # hand = hand.permute(1,2,0).contiguous().numpy() # [64, 64, 3]
        # hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("Data/hand128x128/{}_{}.png".format(name, idx), hand)
        
        # # im = Image.fromarray(img.permute(1,2,0).contiguous().numpy())
        # # im.save("Data/hand128x128/{}_{}.png".format("hand_total", idx))
        # image = img.permute(1,2,0).contiguous().numpy() # [256, 256, 3]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("Data/hand128x128/{}_{}.png".format("hand_total", idx), image)
        # return torch.FloatTensor([miny, maxy, minx, maxx])
        return miny, maxy, minx, maxx
        

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
    data_path = "Data/how2sign"
    class Option:
        hand_generator = True
        resolution = 256
        csv_path=csv_path
        data_path=data_path
        batchSize=12
        num_workers=32
    opts= Option()

    dataloader = How2SignImagePairData(opts).test_dataloader()
    # dataloader = ImagePairDataset(opts)

    for i, data in enumerate(dataloader):
        # if i > 20: break
        print(data["label"].shape)
        print(data["rgb"].shape)
        print(data["lhand_coords"])
    exit()
