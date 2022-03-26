from turtle import forward
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from modules.st_gcn import ConvTemporalGraphical, Graph
import torch.distributed as dist
import argparse
from modules.utils import shift_dim
import numpy as np
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.attention import Transformer, FeedForward
from modules.nearby_attn import AttnBlock
from modules.vq_fn import Codebook
import einops
from modules.sp_layer import SPL
from util.plot_videos import draw_frame_2D


def zero(x):
    return 0

def iden(x):
    return x


class PoseVitVQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.pose_emb = nn.Linear(24, 256)
        self.rhand_emb = nn.Linear(63, 256)
        self.lhand_emb = nn.Linear(63, 256)

        self.enc_vit = Transformer(dim=256, depth=3, heads=8, dim_head=64, mlp_dim=1024, dropout = 0.1)
        self.codebook = Codebook(n_codes=args.n_codes, embedding_dim=256)

        self.dec_vit = Transformer(dim=256, depth=3, heads=8, dim_head=64, mlp_dim=1024, dropout = 0.1)

        self.pose_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=3, reuse=False, sparse=False, SKELETON="sign_pose")        
        self.hand_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=3, reuse=False, sparse=False, SKELETON="sign_hand")        
        
        self.save_hyperparameters()


    def encode(self, pose, rhand, lhand):
        """points: [bs, 150]
        """
        pose = self.pose_emb(pose) # [bs, 512]
        rhand = self.rhand_emb(rhand) # [bs, 512]
        lhand = self.lhand_emb(lhand) # [bs, 512]

        x = torch.cat([pose.unsqueeze(-2), rhand.unsqueeze(-2), lhand.unsqueeze(-2)], dim=-2)  # [bs, 3, h]
        x = self.enc_vit(x)
        x = einops.rearrange(x, "b n h -> b h n")
        vq_output = self.codebook(x)
        
        return vq_output['encodings'], vq_output['embeddings'], vq_output["commitment_loss"] # [b, n], [b,emb_dim,n]

    def decode(self, x):
        """x: [bs, h, n]
        """
        b, h, n = x.size()
        x = einops.rearrange(x, "b h n-> b n h")
        x = self.enc_vit(x)  # [b n h]
        pose = self.pose_spl(x[:, 0, :])  # [b, h] -> [b, 24]
        rhand = self.hand_spl(x[:, 1, :]) # [b, h] -> [b, 63]
        lhand = self.hand_spl(x[:, 2, :]) # [b, h] -> [b, 63]

        return pose, rhand, lhand

    def forward(self, batch, mode):
        points = batch["skel_3d"]  # [bs, 150]
        pose = points[:, :24]
        rhand = points[:, 24:24+63]
        lhand = points[:, 87:150]

        _, feat, commitment_loss = self.encode(pose, rhand, lhand)
        
        dec_pose, dec_rhand, dec_lhand = self.decode(feat)

        pose_rec_loss = torch.abs(pose - dec_pose).mean()
        rhand_rec_loss = torch.abs(rhand - dec_rhand).mean()
        lhand_rec_loss = torch.abs(lhand - dec_lhand).mean()

        rec_loss = pose_rec_loss + rhand_rec_loss + lhand_rec_loss
        loss = rec_loss + commitment_loss

        self.log('{}/commitment_loss'.format(mode), commitment_loss.detach(), prog_bar=True)
        self.log('{}/pose_rec_loss'.format(mode), pose_rec_loss.detach(), prog_bar=True)
        self.log('{}/rhand_rec_loss'.format(mode), rhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/lhand_rec_loss'.format(mode), lhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/rec_loss'.format(mode), rec_loss.detach(), prog_bar=True)
        
        if mode == "train" and self.global_step % 200 == 0:
            self.vis(pose, rhand, lhand, "train", "ori_vis")
            self.vis(dec_pose, dec_rhand, dec_lhand, "train", "dec_vis")

        return {"loss":loss, 
                "origin": [pose, rhand, lhand],
                "prediction": [dec_pose, dec_rhand, dec_lhand]}

    def vis(self, pose, rhand, lhand, mode, name):
        points = torch.cat([pose, rhand, lhand], dim=-1).detach().cpu().numpy()
        # points: [bs, 150]
        show_img = []
        for j in range(4):
            frame_joints = points[j]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d)
            show_img.append(im)
        show_img = np.concatenate(show_img, axis=1) # [h, w, c]
        show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
        show_img = torchvision.utils.make_grid(show_img, )
        self.logger.experiment.add_image("{}/{}".format(mode, name), show_img, self.global_step)
        

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, "train")
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.forward(batch, "val")
        if batch_idx < 2:
            pose, rhand, lhand = out["origin"]
            dec_pose, dec_rhand, dec_lhand = out["prediction"]
            self.vis(pose, rhand, lhand, "val", "ori_vis")
            self.vis(dec_pose, dec_rhand, dec_lhand, "val", "dec_vis")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96, last_epoch=-1)
        return [optimizer], [scheduler]

        # optimizer = torch.optim.Adam(self.parameters(), lr=4e-6, betas=(0.9, 0.999))
        # return [optimizer]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=256)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        return parser


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        x = F.pad(x, self.pad_input)
        return self.convt(x)

class SamePadConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 1
        if isinstance(stride, int):
            stride = (stride,) * 1

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.convt = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        x = F.pad(x, self.pad_input)
        return self.convt(x)





if __name__ == "__main__":
    # N, C, T, V = 5, 2, 16, 25
    # x = torch.randn(N, C, T, V)

    # model = ST_GCN_18(2, 10, {"layout": 'ntu-rgb+d', "strategy": 'spatial'}) 
    # print(model)

    # out = model.extract_feature(x)
    # print(out.shape)
    x = torch.randn(2, 256, 16, 1)

    # m1 = SamePadConv2d(64, 64, 4, (1,2))
    # x = m1(x)
    # print("x: ", x.shape)
    m0 = nn.ConvTranspose2d(256, 64, kernel_size=(1, 3), stride=1)
    # m1 = SamePadConvTranspose2d(64, 64, kernel_size=(1,4), stride=(1,2))
    # m2 = SamePadConvTranspose2d(64, 64, kernel_size=(1,4), stride=(1,2))
    # m3 = nn.ConvTranspose2d(64, 64, kernel_size=(1,5), stride=1) # (input - 1) * stride + output_padding - 2*padding + kernel
    x = m0(x)
    print("out: ", x.shape)
    # x = m1(x)
    # print("out: ", x.shape)
    # x = m2(x)
    # print("out: ", x.shape)
    # x = m3(x)
    # print("out: ", x.shape)

    # print("x: ", x.shape)

    # graph = Graph(layout= "pose25", strategy="spatial")
    # A = torch.tensor(graph.A,
    #                 dtype=torch.float32,
    #                 requires_grad=False)

    # # build networks
    # spatial_kernel_size = A.size(0)
    # print(spatial_kernel_size, A.size())
    # kernel_size = (9, spatial_kernel_size)
    # st_gcn = st_gcn_block(64, 64, kernel_size, 2)

    # x, A = st_gcn(x, A)
    # print("stgcn: ", x.shape)

    # gcn_transtcn = gcn_transtcn_block(64, 64, (4, 3), 2)
    # x, A = gcn_transtcn(x, A)
    # print("trans tcn: ", x.shape)