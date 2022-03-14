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
from modules.attention import Transformer
from modules.nearby_attn import AttnBlock
from modules.vq_fn import Codebook
import einops


def zero(x):
    return 0

def iden(x):
    return x


class Layernorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = 1e-12

    def forward(self, x):
        """x: [bs, c, t, v]
        """
        x = einops.rearrange(x, "b c t v -> b t v c")
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        x = einops.rearrange(x, "b t v c -> b c t v")
        return x


class PoseVitVQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        
        self.linear_emb = nn.Linear(100, 512)

        # downsample
        # step 1
        self.sp1_pose_conv1 = nn.Conv2d(2, 16, kernel_size=(1,4), stride=(1,4))
        self.sp1_pose_conv2 = nn.Conv2d(2, 16, kernel_size=1, stride=1)
        self.sp1_hand_conv1 = nn.Conv2d(2, 16, kernel_size=(1,3), stride=(1,3))
        self.sp1_hand_conv2 = nn.Conv2d(2, 16, kernel_size=1, stride=1)
        self.sp1_norm = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU()
        # step 2
        self.sp2_pose_conv1  = nn.Conv2d(16, 64, kernel_size=(1,3), stride=(1,3))
        self.sp2_pose_conv2  = nn.Conv2d(16, 64, kernel_size=1, stride=1)
        self.sp2_hand_conv1  = nn.Conv2d(16, 64, kernel_size=(1,2), stride=(1,2))
        self.sp2_hand_conv2  = nn.Conv2d(16, 64, kernel_size=1, stride=1)
        self.sp2_norm = nn.BatchNorm2d(64)
        # step 3
        self.sp3_conv1 = nn.Conv2d(64, 256, kernel_size=(1, 6), stride=(1,6))
        self.sp3_conv2 = nn.Conv2d(64, 256, kernel_size=1, stride=1)
        self.sp3_norm = nn.BatchNorm2d(256)
        # step 4
        self.sp4_conv1 = nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(1,3))
        self.sp4_norm = nn.BatchNorm2d(512)

        self.enc_vit = Transformer(dim=512, depth=3, heads=8, dim_head=64, mlp_dim=2048, dropout = 0.1)
        self.pre_vq = nn.Conv1d(512, 512, 1, 1)

        self.post_vq = nn.Conv1d(512, 512, 1, 1)
        self.dec_vit = Transformer(dim=512, depth=3, heads=8, dim_head=64, mlp_dim=2048, dropout = 0.1)
        self.codebook = Codebook(n_codes=5120, embedding_dim=512)
        
        # upsample
        # step 4
        self.sp4_up_norm = nn.BatchNorm2d(512)
        self.sp4_transconv1 = nn.ConvTranspose2d(512, 256, kernel_size=(1,3), stride=1)
        
        # step 3
        self.sp3_up_norm = nn.BatchNorm2d(256)
        self.sp3_transconv1 = nn.ConvTranspose2d(256, 64, kernel_size=(1,6), stride=1)
        self.sp3_transconv2 = nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1)
        
        # step 2
        self.sp2_up_norm = nn.BatchNorm2d(64)
        self.sp2_hand_transconv1 = nn.ConvTranspose2d(64, 16, kernel_size=(1,2), stride=(1,2))
        self.sp2_hand_transconv2 = nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1)
        self.sp2_pose_transconv1 = nn.ConvTranspose2d(64, 16, kernel_size=(1,3), stride=1)
        self.sp2_pose_transconv2 = nn.ConvTranspose2d(64, 16, kernel_size=1, stride=1)
        

        # step 1
        self.sp1_up_norm = nn.BatchNorm2d(16)
        self.sp1_pose_transconv1 = nn.ConvTranspose2d(16, 2, kernel_size=(1,4), stride=1)
        self.sp1_pose_transconv2 = nn.ConvTranspose2d(16, 2, kernel_size=1, stride=1)
        self.sp1_hand_transconv1 = nn.ConvTranspose2d(16, 2, kernel_size=(1,3), stride=(1,3))
        self.sp1_hand_transconv2 = nn.ConvTranspose2d(16, 2, kernel_size=1, stride=1)
        
        self.save_hyperparameters()


    def heuristic_downsample(self, pose, rhand, lhand):
        # step1:
        sp1_pose1 = self.sp1_pose_conv1(pose[..., [0,1,2,5]]) # [bs, 16, t, 1] (0,1,2,5)
        sp1_pose2 = self.sp1_pose_conv2(pose[..., [3,4,6,7]]) # [bs, 16, t, 4], 3,4,6,7

        sp1_rhand1 = self.sp1_hand_conv1(rhand[..., [2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20]])  # [bs, 16, t, 5]
        sp1_rhand2 = self.sp1_hand_conv2(rhand[..., [1,5,9,13,17]])
        
        sp1_lhand1 = self.sp1_hand_conv1(lhand[..., [2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20]]) 
        sp1_lhand2 = self.sp1_hand_conv2(lhand[..., [1,5,9,13,17]])  # [0,1,5,9,13,17]

        sp1_points = torch.cat([sp1_pose1, sp1_pose2, sp1_rhand1, sp1_rhand2, sp1_lhand1, sp1_lhand2], dim=-1)
        sp1_points = self.relu(self.sp1_norm(sp1_points))
        # print("sp1_points: ", sp1_points.shape)
        
        sp1_pose = sp1_points[..., 0:5]
        sp1_rhand = sp1_points[..., 5:15]
        sp1_lhand = sp1_points[..., 15:25]

        # step2
        sp2_pose1 = self.sp2_pose_conv1(sp1_pose[..., [0, 1, 3]]) # [b,c,t,1], (0,1,2,5),3,6
        sp2_pose2 = self.sp2_pose_conv2(sp1_pose[..., [2,4]])  # [b,c,t,2], [4, 7]

        sp2_rhand1 = self.sp2_hand_conv1(sp1_rhand[..., [0,5]]) # [b,c,t,1], (2,3,4), 1
        sp2_rhand2 = self.sp2_hand_conv1(sp1_rhand[..., [1,6]]) # [b,c,t,1], (6,7,8), 5
        sp2_rhand3 = self.sp2_hand_conv1(sp1_rhand[..., [2,7]]) # [b,c,t,1], (10,11,12), 9
        sp2_rhand4 = self.sp2_hand_conv1(sp1_rhand[..., [3,8]]) # [b,c,t,1], (14,15,16), 13
        sp2_rhand5 = self.sp2_hand_conv1(sp1_rhand[..., [4,9]]) # [b,c,t,1], (18,19,20), 17

        sp2_lhand1 = self.sp2_hand_conv1(sp1_lhand[..., [0,5]]) # [b,c,t,1], (2,3,4), 1
        sp2_lhand2 = self.sp2_hand_conv1(sp1_lhand[..., [1,6]]) # [b,c,t,1], (6,7,8), 5
        sp2_lhand3 = self.sp2_hand_conv1(sp1_lhand[..., [2,7]]) # [b,c,t,1], (10,11,12), 9
        sp2_lhand4 = self.sp2_hand_conv1(sp1_lhand[..., [3,8]]) # [b,c,t,1], (14,15,16), 13
        sp2_lhand5 = self.sp2_hand_conv1(sp1_lhand[..., [4,9]]) # [b,c,t,1], (18,19,20), 17

        sp2_points = torch.cat([sp2_pose1, sp2_pose2, sp2_rhand1, sp2_rhand2, sp2_rhand3, sp2_rhand4, sp2_rhand5,
                                sp2_lhand1, sp2_lhand2, sp2_lhand3, sp2_lhand4, sp2_lhand5], dim=-1)
        sp2_points = self.relu(self.sp2_norm(sp2_points))
        # print("sp2_points: ", sp2_points.shape)

        # step 3
        # sp3_rhand = torch.cat([sp2_pose2[..., 0:1], sp2_rhand1, sp2_rhand2, sp2_rhand3, sp2_rhand4, sp2_rhand5], dim=-1) # [b h t 6]
        sp3_rhand = self.sp3_conv1(sp2_points[..., [1, 3,4,5,6,7]])
        # sp3_lhand = torch.cat([sp2_pose2[..., 1:2], sp2_lhand1, sp2_lhand2, sp2_lhand3, sp2_lhand4, sp2_lhand5], dim=-1)
        sp3_lhand = self.sp3_conv1(sp2_points[..., [2, 8,9,10,11,12]])
        sp3_pose = self.sp3_conv2(sp2_points[..., 0:1])
        sp3_point = torch.cat([sp3_rhand, sp3_pose, sp3_lhand], dim=-1)
        sp3_point = self.relu(self.sp3_norm(sp3_point))
        # print("sp3_point: ", sp3_point.shape)
        
        # step 4
        sp4_point = self.sp4_conv1(sp3_point)
        # print("sp4_point: ", sp4_point.shape)

        return sp4_point.squeeze(-1)

    def encode(self, pose, rhand, lhand):
        points_feat = self.heuristic_downsample(pose, rhand, lhand)
        points_feat = self.enc_vit(einops.rearrange(points_feat, "b c t -> b t c"))
        points_feat = einops.rearrange(points_feat, "b t c -> b c t")
        vq_output = self.codebook(self.pre_vq(points_feat))
        
        return vq_output['encodings'], vq_output['embeddings'], vq_output["commitment_loss"]

    def decode(self, feat):
        feat = self.post_vq(feat)
        feat = self.dec_vit(einops.rearrange(feat, "b c t -> b t c"))
        feat = einops.rearrange(feat, "b t c -> b c t")
        feat = feat.unsqueeze(-1)  # b c t, 1

        # step 4
        feat = self.relu(feat)
        feat = self.sp4_up_norm(feat)
        sp4_feat = self.sp4_transconv1(feat) # b c t, 3
        # print("sp4_feat: ", sp4_feat.shape)

        # step 3
        sp4_feat = self.relu(sp4_feat)
        sp4_feat = self.sp3_up_norm(sp4_feat)
        sp3_rhand = self.sp3_transconv1(sp4_feat[:, :, :, 0:1]) # b c t, 6
        sp3_pose = self.sp3_transconv2(sp4_feat[:, :, :, 1:2]) # b c t, 1
        sp3_lhand = self.sp3_transconv1(sp4_feat[:, :, :, 2:3]) # b c t, 6

        # step 2
        sp2_rhand = self.sp2_hand_transconv1(sp3_rhand[..., 1:6]) # [b, c, t 10]
        sp2_lhand = self.sp2_hand_transconv1(sp3_lhand[..., 1:6]) # [b, c, t 10]
        sp2_pose1 = self.sp2_pose_transconv1(sp3_pose) # [b, c, t, 3]
        sp2_pose2 = self.sp2_pose_transconv2(torch.cat([sp3_rhand[..., 0:1], sp3_lhand[..., 0:1]], dim=-1)) # [bs, c, t, 2], [4,7]

        # step 1
        sp1_pose1 = self.sp1_pose_transconv1(sp2_pose1[..., 1:2]) # [b, c, t, 4] node for (0,1,2,5)
        sp1_pose2 = self.sp1_pose_transconv2(torch.cat([sp2_pose1[..., [0,2]], sp2_pose2], dim=-1)) # [bs, c, t, 4] node for (3,6,4,7)

        sp1_rhand1 = self.sp1_hand_transconv1(sp2_rhand[..., [1,3,5,7,9]]) # [b, c, t, 15] (2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20)
        sp1_rhand2 = self.sp1_hand_transconv2(sp2_rhand[..., [0,2,4,6,8]]) # [b ,c, t, 5]  (1, 5, 9, 13, 17)

        sp1_lhand1 = self.sp1_hand_transconv1(sp2_lhand[..., [1,3,5,7,9]]) # [b, c, t, 15] (2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20)
        sp1_lhand2 = self.sp1_hand_transconv2(sp2_lhand[..., [0,2,4,6,8]]) # [b ,c, t, 5]

        
        dec_pose = torch.cat([sp1_pose1, sp1_pose2], dim=-1) # 0,1,2,5,3,6,4,7
        dec_rhand = torch.cat([sp1_rhand1, sp1_rhand2], dim=-1) # 2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20, 1, 5, 9, 13, 17
        dec_lhand = torch.cat([sp1_lhand1, sp1_lhand2], dim=-1) # 2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20, 1, 5, 9, 13, 17
        
        pose_idx = {}
        for i, idx in enumerate([0,1,2,5,3,6,4,7]):
            pose_idx[idx] = i
        pose_idx = sorted(pose_idx.items(), key=lambda item : item[0], reverse=False)
        pose_idx = [item[1] for item in pose_idx]
        # print("pose_idx: ", pose_idx)

        hand_idx = {}
        for i, idx in enumerate([2,3,4, 6,7,8, 10,11,12, 14,15,16, 18,19,20, 1, 5, 9, 13, 17]):
            hand_idx[idx] = i
        hand_idx = sorted(hand_idx.items(), key=lambda item : item[0], reverse=False)
        hand_idx = [item[1] for item in hand_idx]
        # print("hand_idx: ", hand_idx)

        dec_pose = dec_pose[..., pose_idx] # [b, c, t, 8]
        dec_rhand = torch.cat([dec_rhand[..., hand_idx], dec_pose[..., 4:5]], dim=-1) # [b, c, t, 21]
        dec_lhand = torch.cat([dec_lhand[..., hand_idx], dec_pose[..., 7:8]], dim=-1) # [b, c, t, 21]

        return dec_pose, dec_rhand, dec_lhand


    def forward(self, batch, mode):
        pose = batch["pose"][..., list(range(8))]
        rhand = batch["rhand"]
        lhand = batch["lhand"]

        _, feat, commitment_loss = self.encode(pose, rhand, lhand)
        dec_pose, dec_rhand, dec_lhand = self.decode(feat)
        
        
        
        pose_no_mask = batch["pose_no_mask"][..., list(range(8))]
        rhand_no_mask = batch["rhand_no_mask"]
        lhand_no_mask = batch["lhand_no_mask"]


        pose_rec_loss = (torch.abs(pose - dec_pose) * pose_no_mask).sum() / (pose_no_mask.sum() + 1e-7)
        rhand_rec_loss = (torch.abs(rhand - dec_rhand) * rhand_no_mask).sum() / (rhand_no_mask.sum()+ 1e-7)
        lhand_rec_loss = (torch.abs(lhand - dec_lhand) * lhand_no_mask).sum() / (lhand_no_mask.sum() + 1e-7)

        loss = pose_rec_loss + rhand_rec_loss + lhand_rec_loss + commitment_loss

        self.log('{}/commitment_loss'.format(mode), commitment_loss.detach(), prog_bar=True)
        self.log('{}/pose_rec_loss'.format(mode), pose_rec_loss.detach(), prog_bar=True)
        self.log('{}/rhand_rec_loss'.format(mode), rhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/lhand_rec_loss'.format(mode), lhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        for i in range(4):
            if mode == "train" and self.global_step % 200 == 0:
                self.visualization(mode, "orig_vis", pose[i], rhand[i], lhand[i])
                self.visualization(mode, "pred_vis", dec_pose[i], dec_rhand[i], dec_lhand[i])
            
        return {"loss":loss, 
                "origin": [pose, rhand, lhand],
                "prediction": [dec_pose, dec_rhand, dec_lhand]}

    def training_step(self, batch, batch_idx):
        out = self.forward(batch, "train")
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        out = self.forward(batch, "val")
        if batch_idx < 2:
            pose, rhand, lhand = out["origin"]
            dec_pose, dec_rhand, dec_lhand = out["prediction"]
            for i in range(4):
                self.visualization("val", "orig_vis", pose[i], rhand[i], lhand[i])
                self.visualization("val", "pred_vis", dec_pose[i], dec_rhand[i], dec_lhand[i])

    def visualization(self, mode, name, pose, rhand, lhand, log=True):
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
        if log:
            self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)
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

class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 ds_kernels,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, ds_kernels[0], **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, ds_kernels[1], **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, c, t, v)


        return feature


class GCN_TranTCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 1
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            gcn_transtcn_block(256, 256, kernel_size, 1, residual=False, **kwargs0),
            gcn_transtcn_block(256, 256, kernel_size, 1, **kwargs),
            gcn_transtcn_block(256, 128, kernel_size, 1, **kwargs),
            gcn_transtcn_block(128, 128, kernel_size, 1, **kwargs),
            gcn_transtcn_block(128, 128, kernel_size, 1, **kwargs),
            gcn_transtcn_block(128, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 2, kernel_size, 1, act_fn=nn.Tanh(), **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        print("after encoer: ", x.shape)
        exit()
        return x




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