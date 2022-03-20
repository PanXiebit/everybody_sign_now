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

def zero(x):
    return 0

def iden(x):
    return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, scale_factor):
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class PoseVitVQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.pose_emb = nn.Linear(16, 256)
        self.hand_emb = nn.Linear(42, 256)

        # self.enc_vit = Transformer(dim=256, depth=3, heads=8, dim_head=64, mlp_dim=1024, dropout = 0.1)
        self.enc_ffn = FeedForward(256*3, 256*6)
        self.codebook = Codebook(n_codes=5120, embedding_dim=256*3)

        self.dec_ffn = FeedForward(256*3, 256*6)

        self.pose_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=2, reuse=False, sparse=False, SKELETON="sign_pose")        
        self.hand_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=2, reuse=False, sparse=False, SKELETON="sign_hand")        
        
        self.save_hyperparameters()


    def encode(self, pose, rhand, lhand):
        """pose: [bs, 2, t, 8]
           rhand: [bs, 2, t, 21]
           lhand: [bs, 2, t, 21]
        """
        pose = einops.rearrange(pose, "b c t v -> b t (c v)")
        pose = self.pose_emb(pose) # [bs, t, 1, h]
        rhand = einops.rearrange(rhand, "b c t v -> b t (c v)")
        rhand = self.hand_emb(rhand) # [bs, t, 1, h]
        lhand = einops.rearrange(lhand, "b c t v -> b t (c v)")
        lhand = self.hand_emb(lhand) # [bs, t, 1, h]

        x = torch.cat([pose.unsqueeze(-2), rhand.unsqueeze(-2), lhand.unsqueeze(-2)], dim=-2)  # [bs, t, 3, h]
        x = einops.rearrange(x, "b t n h -> b t (n h)") # [bs, t*3, h]
        x = self.enc_ffn(x)
        x = einops.rearrange(x, "b t h -> b h t")
        vq_output = self.codebook(x)
        return vq_output['encodings'], vq_output['embeddings'], vq_output["commitment_loss"]

    def decode(self, x):
        """x: [bs, c, t]
        """
        b, _, t = x.size()
        x = einops.rearrange(x, "b h t-> b t h")
        x = self.dec_ffn(x)
        x = einops.rearrange(x, "b t (n h) -> (b t) n h", n=3)
        pose = self.pose_spl(x[:, 0, :])
        rhand = self.hand_spl(x[:, 1, :])
        lhand = self.hand_spl(x[:, 2, :])
        pose = einops.rearrange(pose, "(b t) (c v) -> b c t v", b=b, c=2, v=8)
        rhand = einops.rearrange(rhand, "(b t) (c v) -> b c t v", b=b, c=2, v=21)
        lhand = einops.rearrange(lhand, "(b t) (c v) -> b c t v", b=b, c=2, v=21)
        
        return pose, rhand, lhand

    def forward(self, batch, mode):
        pose = batch["pose"][..., [1,0,2,3,4,5,6,7]]
        rhand = batch["rhand"]
        lhand = batch["lhand"]
        _, feat, commitment_loss = self.encode(pose, rhand, lhand)
        
        dec_pose, dec_rhand, dec_lhand = self.decode(feat)
        
        pose_no_mask = batch["pose_no_mask"][..., [1,0,2,3,4,5,6,7]]
        rhand_no_mask = batch["rhand_no_mask"]
        lhand_no_mask = batch["lhand_no_mask"]

        pose_rec_loss = (torch.abs(pose - dec_pose) * pose_no_mask).sum() / (pose_no_mask.sum() + 1e-7)
        rhand_rec_loss = (torch.abs(rhand - dec_rhand) * rhand_no_mask).sum() / (rhand_no_mask.sum()+ 1e-7)
        lhand_rec_loss = (torch.abs(lhand - dec_lhand) * lhand_no_mask).sum() / (lhand_no_mask.sum() + 1e-7)

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
        pose = pose[..., [1,0,2,3,4,5,6,7,]]
        for i in range(4):
            self.visualization(mode, name, pose[i], rhand[i], lhand[i], i)
            

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

    def visualization(self, mode, name, pose, rhand, lhand, idx):
        # visualize
        ori_vis = []

        c, t, v = pose.size()
        pose = pose.permute(1, 2, 0).contiguous()  # [t, v, c]
        rhand = rhand.permute(1, 2, 0).contiguous()
        lhand = lhand.permute(1, 2, 0).contiguous()

        for i in range(pose.size(0)):   
            pose_anchor = (640, 360)
            pose_list = self._tensor2numpy(pose[i], pose_anchor, "pose", 25, list(range(8))) # [3V]


            rhand_list = self._tensor2numpy(rhand[i], pose_anchor, "rhand", 21, list(range(21)))# , rhand_anchor[0] * 640, rhand_anchor[1] * 360)
            lhand_list = self._tensor2numpy(lhand[i], pose_anchor, "lhand", 21, list(range(21))) # , lhand_anchor[0] * 640, lhand_anchor[1] * 360)

            canvas = self._render(pose_list, rhand_list, lhand_list)
            canvas = torch.FloatTensor(canvas) # [h, w, c]
            canvas = canvas.permute(2, 0, 1).contiguous().unsqueeze(0)
            
            ori_vis.append(canvas) # [1, c, h, w]
        ori_vis = torch.cat(ori_vis, dim=0)
        ori_vis = torchvision.utils.make_grid(ori_vis, )
        self.logger.experiment.add_image("{}/{}_{}".format(mode, name, idx), ori_vis, self.global_step)
        return mode, name, ori_vis
    
    def _tensor2numpy(self, points, anchor, part_name, keypoint_num, pose_tokens):
        """[v, c]]
        """
        points = points.detach().cpu().numpy()
        v, c = points.shape
        # [[17, 15, 0, 16, 18], [0, 1, 8, 9, 12], [4, 3, 2, 1, 5], [2, 1, 5, 6, 7]]
        # pose_tokens = []
        assert points.shape[0] == len(pose_tokens)

        pose_vis = np.zeros((keypoint_num, 3), dtype=np.float32)
        for i in range(len(pose_tokens)):
            pose_vis[pose_tokens[i], 0] = points[i][0] * 1280 + anchor[0]
            pose_vis[pose_tokens[i], 1] = points[i][1] * 720 + anchor[1]
            pose_vis[pose_tokens[i], -1] = 1.
        
        pose_vis = pose_vis.reshape((-1, ))
        # print(pose_vis.shape, pose_vis)
        return pose_vis.tolist()


    def _render(self, posepts, r_handpts, l_handpts):
        myshape = (720, 1280, 3)
        numkeypoints = 70
        canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
        canvas = renderhand(r_handpts, canvas)
        canvas = renderhand(l_handpts, canvas) # [720, 720, 3]
        canvas = canvas[:, 280:1000, :]
        canvas = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC) # [256, 256, 3]

        # img = Image.fromarray(canvas[:, :, [2,1,0]])

        return canvas # [256, 256, 3]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
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