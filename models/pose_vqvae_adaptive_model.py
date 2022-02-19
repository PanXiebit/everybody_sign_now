from multiprocessing.sharedctypes import Value
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

def zero(x):
    return 0

def iden(x):
    return x

class PoseVitVQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.temporal_downsample == 4:
            ds_kernels = [2,2]
        elif args.temporal_downsample == 2:
            ds_kernels = [1,2]
        elif args.temporal_downsample == 1:
            ds_kernels = [1,1]
        
        self.st_gcn = nn.ModuleDict()
        self.st_gcn["pose"] = ST_GCN_18(in_channels=2, ds_kernels=ds_kernels, graph_cfg={'layout':'pose25', 'strategy':'spatial'})
        self.st_gcn["face"] = ST_GCN_18(in_channels=2, ds_kernels=ds_kernels, graph_cfg={'layout':'face70', 'strategy':'spatial'})
        self.st_gcn["hand"] = ST_GCN_18(in_channels=2, ds_kernels=ds_kernels, graph_cfg={'layout':'hand21', 'strategy':'spatial'}) 
        self.tokens = {} # 18, 21, 21, 24
        self.tokens["pose"] = list(range(18)) # 18
        self.tokens["face"] = list(range(0, 70, 3)) # 24
        self.tokens["rhand"] = list(range(21)) # 21
        self.tokens["lhand"] = list(range(21)) # 21

        # encode transformer
        if args.atten_type == "spatial-temporal-joint":
            self.st_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        elif args.atten_type == "spatial-temporal-divided":
            self.s_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
            self.t_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        elif args.atten_type == "spatial-only":
            self.s_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        else:
            raise ValueError("attn_type is wrong!")
        
        # downsample
        self.downsample = Encoder(256, [1,4])
        self.pre_vq_conv = SamePadConv2d(256, 256, 1)
        
        
        self.pose_codebook = Codebook(args.n_codes, args.embedding_dim)
        self.face_codebook = Codebook(args.n_codes, args.embedding_dim)
        self.rhand_codebook = Codebook(args.n_codes, args.embedding_dim)
        self.lhand_codebook = Codebook(args.n_codes, args.embedding_dim)
        
        # dec transformer
        self.post_vq_conv = SamePadConv2d(256, 256, 1)

        if args.atten_type == "spatial-temporal-joint":
            self.dec_st_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        elif args.atten_type == "spatial-temporal-divided":
            self.dec_s_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
            self.dec_t_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        elif args.atten_type == "spatial-only":
            self.dec_s_transformer = Transformer(dim=256, depth=3, heads=4, dim_head=64, mlp_dim=1024)
        else:
            raise ValueError("attn_type is wrong!")

        # upsample
        self.upsample = nn.ModuleDict()
        self.upsample["pose"] = Decoder(256, [1,4], "pose")
        self.upsample["face"] = Decoder(256, [1,4], "face")
        self.upsample["rhand"] = Decoder(256, [1,4], "rhand")
        self.upsample["lhand"] = Decoder(256, [1,4], "lhand")

        self.save_hyperparameters()

    def enc_transformer(self, feat):
        bs, h, t, v = feat.size()
        if self.args.atten_type == "spatial-temporal-joint":
            feat = feat.view(bs, h, t*v).permute(0, 2, 1).contiguous()
            feat = self.st_transformer(feat).permute(0, 2, 1).contiguous()  # [bs, hidden, t/4, 20]
            feat = feat.view(bs, h, t, v)
        elif self.args.atten_type == "spatial-temporal-divided":
            feat = feat.permute(0, 2, 3, 1).contiguous().view(bs*t, v, h)  # [bs, h, t, v] -> [bs, t, v, h] -> [bs*t, v, h]
            feat = self.s_transformer(feat) # [bs*t, v, h]
            feat = feat.view(bs, t, v, h).permute(0, 2, 1, 3).contiguous()  # [bs, v, t, h]
            feat = feat.view(bs*v, t, h)
            feat = self.t_transformer(feat).view(bs, v, t, h)
            feat = feat.permute(0, 3, 2, 1).contiguous() # [bs, h, t, v]
        elif self.args.atten_type == "spatial-only":
            feat = feat.permute(0, 2, 3, 1).contiguous().view(bs*t, v, h)
            feat = self.s_transformer(feat) # [bs*t, v, h]
            feat = feat.view(bs, t, v, h).permute(0, 3, 1, 2).contiguous() # [bs, h, t, v]
        else:
            raise ValueError("attn_type is wrong!")
        return feat

    def dec_transformer(self, feat):
        bs, h, t, v = feat.size()
        if self.args.atten_type == "spatial-temporal-joint":
            feat = feat.view(bs, h, t*v).permute(0, 2, 1).contiguous()
            feat = self.dec_st_transformer(feat).permute(0, 2, 1).contiguous()  # [bs, hidden, t/4, 20]
            feat = feat.view(bs, h, t, v)
        elif self.args.atten_type == "spatial-temporal-divided":
            feat = feat.permute(0, 2, 3, 1).contiguous().view(bs*t, v, h)  # [bs, h, t, v] -> [bs, t, v, h] -> [bs*t, v, h]
            feat = self.dec_s_transformer(feat) # [bs*t, v, h]
            feat = feat.view(bs, t, v, h).permute(0, 2, 1, 3).contiguous()  # [bs, v, t, h]
            feat = feat.view(bs*v, t, h)
            feat = self.dec_t_transformer(feat).view(bs, v, t, h)
            feat = feat.permute(0, 3, 2, 1).contiguous() # [bs, h, t, v]
        elif self.args.atten_type == "spatial-only":
            feat = feat.permute(0, 2, 3, 1).contiguous().view(bs*t, v, h)
            feat = self.dec_s_transformer(feat) # [bs*t, v, h]
            feat = feat.view(bs, t, v, h).permute(0, 3, 1, 2).contiguous() # [bs, h, t, v]
        else:
            raise ValueError("attn_type is wrong!")
        return feat

    def encode(self, batch):
        pose_feat = self.st_gcn["pose"](batch["pose"])[:, :, :, self.tokens["pose"]]
        face_feat = self.st_gcn["face"](batch["face"])[:, :, :, self.tokens["face"]]
        rhand_feat = self.st_gcn["hand"](batch["rhand"])
        lhand_feat = self.st_gcn["hand"](batch["lhand"])
        
        # print("pose_feat: ", pose_feat.shape, face_feat.shape, rhand_feat.shape, lhand_feat.shape)
        pose_feat = self.enc_transformer(pose_feat)
        face_feat = self.enc_transformer(face_feat)
        rhand_feat = self.enc_transformer(rhand_feat)
        lhand_feat = self.enc_transformer(lhand_feat)
        # print("pose_feat: ", pose_feat.shape, face_feat.shape, rhand_feat.shape, lhand_feat.shape)

        # downsample
        pose_feat = self.downsample(pose_feat)
        face_feat = self.downsample(face_feat)
        rhand_feat = self.downsample(rhand_feat)
        lhand_feat = self.downsample(lhand_feat)

        pose_vq_output = self.pose_codebook(self.pre_vq_conv(pose_feat))
        face_vq_output = self.face_codebook(self.pre_vq_conv(face_feat))
        rhand_vq_output = self.rhand_codebook(self.pre_vq_conv(rhand_feat))
        lhand_vq_output = self.lhand_codebook(self.pre_vq_conv(lhand_feat))
        
        return pose_vq_output, face_vq_output, rhand_vq_output, lhand_vq_output
    
    def decode(self, vq_output, part_name):
        _, feat = vq_output['encodings'], vq_output['embeddings']
        feat = self.dec_transformer(self.post_vq_conv(feat))
        # print("feat: ", feat.shape)
        
        feat = self.upsample[part_name](feat)
        return feat



    def forward(self, batch, mode):
        pose_vq_output, face_vq_output, rhand_vq_output, lhand_vq_output = self.encode(batch)
        pose_pred = self.decode(pose_vq_output, "pose")
        face_pred = self.decode(face_vq_output, "face")
        rhand_pred = self.decode(rhand_vq_output, "rhand")
        lhand_pred = self.decode(lhand_vq_output, "lhand")

        # print("predictions: ", pose_pred.shape, face_pred.shape, rhand_pred.shape, lhand_pred.shape)

        pose = batch["pose"][:, :, :, self.tokens["pose"]]
        face = batch["face"][:, :, :, self.tokens["face"]]
        rhand = batch["rhand"]
        lhand = batch["lhand"]

        
        pose_no_mask = batch["pose_no_mask"][:, :, :, self.tokens["pose"]]
        face_no_mask = batch["face_no_mask"][:, :, :, self.tokens["face"]]
        rhand_no_mask = batch["rhand_no_mask"]
        lhand_no_mask = batch["lhand_no_mask"]

        # print("pose: ", pose.shape, face.shape, rhand.shape, lhand.shape)
        # print("pose_no_mask: ", pose_no_mask.shape, face_no_mask.shape, rhand_no_mask.shape, lhand_no_mask.shape)
        

        pose_rec_loss = (torch.abs(pose - pose_pred) * pose_no_mask).sum() / (pose_no_mask.sum() + 1e-7)
        # print("pose_rec_loss: ", pose_rec_loss.shape, pose_no_mask.shape)
        face_rec_loss = (torch.abs(face - face_pred) * face_no_mask).sum() / (face_no_mask.sum() + 1e-7)
        rhand_rec_loss = (torch.abs(rhand - rhand_pred) * rhand_no_mask).sum() / (rhand_no_mask.sum()+ 1e-7)
        lhand_rec_loss = (torch.abs(lhand - lhand_pred) * lhand_no_mask).sum() / (lhand_no_mask.sum() + 1e-7)

        loss = pose_rec_loss + face_rec_loss + rhand_rec_loss + lhand_rec_loss

        self.log('{}/pose_rec_loss'.format(mode), pose_rec_loss.detach(), prog_bar=True)
        self.log('{}/face_rec_loss'.format(mode), face_rec_loss.detach(), prog_bar=True)
        self.log('{}/rhand_rec_loss'.format(mode), rhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/lhand_rec_loss'.format(mode), lhand_rec_loss.detach(), prog_bar=True)
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        
        if mode == "train" and self.global_step % 200 == 0:
            self.visualization(mode, "orig_vis", pose, face, rhand, lhand)
            self.visualization(mode, "pred_vis", pose_pred, face_pred, rhand_pred, lhand_pred)
        if mode == "val":
            self.visualization(mode, "orig_vis", pose, face, rhand, lhand)
            self.visualization(mode, "pred_vis", pose_pred, face_pred, rhand_pred, lhand_pred)
        
        return loss


    def visualization(self, mode, name, pose, face, rhand, lhand):
        # visualize
        ori_vis = []

        bs, c, t, v = pose.size()
        pose = pose[0].permute(1, 2, 0).contiguous()  # [t, v, c]
        face = face[0].permute(1, 2, 0).contiguous()
        rhand = rhand[0].permute(1, 2, 0).contiguous()
        lhand = lhand[0].permute(1, 2, 0).contiguous()

        for i in range(pose.size(0)):   
            pose_anchor = (640, 360)
            pose_list = self._tensor2numpy(pose[i], pose_anchor, "pose", 25) # [3V]

            face_anchor = (pose_list[0*3], pose_list[0*3 + 1])
            rhand_anchor = (pose_list[4*3], pose_list[4*3 + 1])
            lhand_anchor = (pose_list[7*3], pose_list[7*3 + 1])

            face_list = self._tensor2numpy(face[i], face_anchor, "face", 70) #, face_anchor[0] * 640, face_anchor[1] * 360)
            rhand_list = self._tensor2numpy(rhand[i], rhand_anchor, "rhand", 21)# , rhand_anchor[0] * 640, rhand_anchor[1] * 360)
            lhand_list = self._tensor2numpy(lhand[i], lhand_anchor, "lhand", 21) # , lhand_anchor[0] * 640, lhand_anchor[1] * 360)

            canvas = self._render(pose_list, face_list, rhand_list, lhand_list)
            canvas = torch.FloatTensor(canvas) # [h, w, c]
            canvas = canvas.permute(2, 0, 1).contiguous().unsqueeze(0)
            
            ori_vis.append(canvas) # [1, c, h, w]
        ori_vis = torch.cat(ori_vis, dim=0)
        ori_vis = torchvision.utils.make_grid(ori_vis, )
        self.logger.experiment.add_image("{}/{}".format(mode, name), ori_vis, self.global_step)
        return mode, name, ori_vis
    
    def _tensor2numpy(self, points, anchor, part_name, keypoint_num):
        """[v, c]]
        """
        points = points.detach().cpu().numpy()
        v, c = points.shape
        # [[17, 15, 0, 16, 18], [0, 1, 8, 9, 12], [4, 3, 2, 1, 5], [2, 1, 5, 6, 7]]
        pose_tokens = self.tokens[part_name]
        # for ids in self.tokens[part_name]:
        #     pose_tokens.extend(ids)

        pose_vis = np.zeros((keypoint_num, 3), dtype=np.float32)
        for i in range(len(pose_tokens)):
            pose_vis[pose_tokens[i], 0] = points[i][0] * 1280 + anchor[0]
            pose_vis[pose_tokens[i], 1] = points[i][1] * 720 + anchor[1]
            pose_vis[pose_tokens[i], -1] = 1.
        
        pose_vis = pose_vis.reshape((-1, ))
        # print(pose_vis.shape, pose_vis)
        return pose_vis.tolist()


    def _render(self, posepts, facepts, r_handpts, l_handpts):
        myshape = (720, 1280, 3)
        numkeypoints = 70
        canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
        canvas = renderface_sparse(facepts, canvas, numkeypoints, disp=False)
        canvas = renderhand(r_handpts, canvas)
        canvas = renderhand(l_handpts, canvas) # [720, 720, 3]
        canvas = canvas[:, 280:1000, :]
        canvas = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC) # [256, 256, 3]

        # img = Image.fromarray(canvas[:, :, [2,1,0]])

        return canvas # [256, 256, 3]

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss


    def validation_step(self, batch, batch_idx):
        if batch_idx > 10: return
        self.forward(batch, "val")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5, last_epoch=-1)
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


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv2d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv2d(in_channels, n_hiddens, kernel_size=3)

    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        return h

class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, part_name):
        super().__init__()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 2 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            if i == max_us-1:
                if part_name == "pose":
                    kernel = (4,6)
                elif part_name == "rhand" or part_name == "lhand":
                    kernel = (4,5)
                elif part_name == "face":
                    kernel = (4,4)
                else:
                    raise ValueError("part_name is not existed!")
            else:
                kernel = 4
            convt = SamePadConvTranspose2d(n_hiddens, out_channels, kernel,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, h):
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        
        x = self.tcn(x) + res
        x += res

        return self.relu(x), A


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, v]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # [b*t*v, c]
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings



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
    N, C, T, V = 5, 256, 16, 25
    x = torch.randn(N, C, T, V)
    m = SamePadConv2d(256, 256, kernel_size=4, stride=2)
    m2 = SamePadConvTranspose2d(256, 256, 5, 2)
    x = m2(x)
    print(x.shape)
