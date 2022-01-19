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
        keypoint_folder = os.path.join(data_path, tag, "openpose_output/")

        data = pd.read_csv(csv_path, on_bad_lines='skip', delimiter="\t")
        debug = 5
        warnings.filterwarnings('ignore')

        key_json_files = []
        sentences = []

        for i in tqdm(range(len(data))):
            if debug and i >= debug: break
            if "CTERDLghzFw_7-8-rgb_front" == data["SENTENCE_NAME"][i]: 
                continue
            key_json_path = os.path.join(keypoint_folder, "json", data["SENTENCE_NAME"][i])
            sent = data["SENTENCE"][i].lower().strip().split()
            try:
                assert os.path.exists(key_json_path), "{}".format(key_json_path)
            except:
                # print(data["SENTENCE_NAME"][i])
                continue
            key_json_files.append(key_json_path)
            sentences.append(sent)

        self._sentences = sentences
        self._key_json_files = key_json_files
        assert len(sentences) == len(key_json_files)
        print("{} video number is: {}/{}".format(tag, len(sentences), len(data)))

        
    def __len__(self):
        return len(self._sentences)

    def __getitem__(self, idx):
        sent = self._sentences[idx]
        keypoint_path = self._key_json_files[idx]
        keypoint_files = sorted(os.listdir(keypoint_path))[::2]
        points = []
        for keypoint_file in keypoint_files:
            posepts, facepts, r_handpts, l_handpts = self.readkeypointsfile_json(os.path.join(keypoint_path, keypoint_file))
            # print(len(posepts), len(facepts), len(r_handpts), len(l_handpts))
            all_keypoints = np.array(posepts + facepts + r_handpts + l_handpts, dtype=np.float32) # 25, 
            points.append(np.expand_dims(all_keypoints, axis=0))
        points = np.concatenate(points, axis=0)

        return dict(points=points, sent=sent)

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

    def crop_and_resize(self, images, cropped_shape, resize_shape, hands_points):
        # crop and resize
        y, x, _ = images[0].shape # [720, 1280]

        crop_y, crop_x = cropped_shape # [720, 720]

        shift_y, start_y, end_y = 0, 0, y
        shift_x, start_x, end_x = 0, 0, x
        if crop_y < y:
            shift_y = (y - crop_y) //2
            start_y, end_y = shift_y, shift_y + crop_y
        if crop_x < x:
            shift_x = (x - crop_x) // 2
            start_x, end_x = shift_x, shift_x + crop_x
        images = [img[start_y:end_y, start_x:end_x, :] for img in images]
        img_transforms = transforms.Compose([transforms.Resize(resize_shape)])
        images = [img_transforms(img.permute(2, 0, 1).contiguous()) for img in images] # [3, 256, 256]

        scale = resize_shape[0] / cropped_shape[0]
        hands_points = [fix_scale_coords(points, scale, (-shift_x, -shift_y)) for points in hands_points]
        return images, hands_points


    def _get_hands(self, img, lhand_points, idx, name):
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
        endy, endx = img.size(1), img.size(2) # 720, 1280
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
    dataloader = ImagePairDataset(opts)

    for i, data in enumerate(dataloader):
        if i > 20: break
        print(data["points"].shape)
        print(data["sent"])
    exit()
