import torch
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


def zero(x):
    return 0

def iden(x):
    return x



class PoseVQVAE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.encoder = nn.ModuleDict()
        self.encoder["pose"] = ST_GCN_18(in_channels=2, graph_cfg={'layout':'pose25', 'strategy':'spatial'})
        self.encoder["face"] = ST_GCN_18(in_channels=2, graph_cfg={'layout':'face70', 'strategy':'spatial'})
        self.encoder["hand"] = ST_GCN_18(in_channels=2, graph_cfg={'layout':'hand21', 'strategy':'spatial'})

        self.points_downsample = nn.ModuleList()
        n_times_downsample = 2
        for i in range(n_times_downsample):
            conv = SamePadConv2d(256, 256, 4, stride=(1,2))
            self.points_downsample.append(conv)
        
        self.pre_vq_conv = SamePadConv2d(256, 256, 1)
        self.post_vq_conv = SamePadConv2d(256, 256, 1)
        self.codebook = Codebook(args.n_codes, args.embedding_dim)

        self.face_upsample = nn.ModuleList()
        n_times_downsample = 2
        for i in range(n_times_downsample):
            if i != n_times_downsample-1: kernel =5
            else: kernel=4
            conv = SamePadConvTranspose2d(256, 256, kernel, stride=(1,2))
            self.face_upsample.append(conv)

        self.pose_upsample = nn.ModuleList()
        n_times_downsample = 2
        for i in range(n_times_downsample):
            if i != n_times_downsample-1: kernel = 4
            else: kernel=5
            conv = SamePadConvTranspose2d(256, 256, kernel, stride=(1,2))
            self.pose_upsample.append(conv)

        self.decoder = nn.ModuleDict()
        self.decoder["pose"] = GCN_TranTCN_18(in_channels=256, graph_cfg={'layout':'pose25', 'strategy':'spatial'})
        self.decoder["face"] = GCN_TranTCN_18(in_channels=256, graph_cfg={'layout':'face70', 'strategy':'spatial'})
        self.decoder["hand"] = GCN_TranTCN_18(in_channels=256, graph_cfg={'layout':'hand21', 'strategy':'spatial'})


    def encode(self, points, part_name):
        feat = self.encoder[part_name](points)
        for conv in self.points_downsample:
            feat = conv(feat)
        vq_output = self.codebook(self.pre_vq_conv(feat))

        return vq_output['encodings'], vq_output['embeddings']

    def decode(self, feat, part_name):
        feat = self.post_vq_conv(feat)
        if part_name == "face":
            for tcn in self.face_upsample:
                feat = tcn(feat)
        else:
            for tcn in self.pose_upsample:
                feat = tcn(feat)
        out = self.decoder[part_name](feat)
        return out

    def reconstruction(self, points, part_name):
        recon = self.decode(self.encode(points, part_name)[1], part_name)
        return recon


    def forward(self, batch, mode):
        pose, face, rhand, lhand = batch["pose"], batch["face"], batch["rhand"], batch["lhand"] # [bs, c, t, v]

        pose_no_mask, face_no_mask, rhand_no_mask, lhand_no_mask = batch["pose_no_mask"], batch["face_no_mask"], batch["rhand_no_mask"], batch["lhand_no_mask"]

        pose_pred = self.reconstruction(pose, "pose")
        face_pred = self.reconstruction(face, "face")
        rhand_pred = self.reconstruction(rhand, "hand")
        lhand_pred = self.reconstruction(lhand, "hand")

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

        # visualize
        ori_pose_np = self._tensor2numpy(pose, 640, 360)

        face_anchor = pose[0, :, 0, 0]
        rhand_anchor = pose[0, :, 0, 7]
        lhand_anchor = pose[0, :, 0, 4]
        
        ori_face_np = self._tensor2numpy(face, 640, 360) #, face_anchor[0] * 640, face_anchor[1] * 360)
        ori_rhand_np = self._tensor2numpy(rhand, 640, 360)# , rhand_anchor[0] * 640, rhand_anchor[1] * 360)
        ori_lhand_np = self._tensor2numpy(lhand, 640, 360) # , lhand_anchor[0] * 640, lhand_anchor[1] * 360)


        ori_vis = []
        bs, c, t, v = pose.size()

        for i in range(t):
            pose = pose.permute(0, 2, 3, 1).contiguous() # [bs, t, v, c]

            post_list = ori_pose_np[i].tolist()
            face_list = ori_face_np[i].tolist()
            rhand_list = ori_rhand_np[i].tolist()
            lhand_list = ori_lhand_np[i].tolist()

            canvas = self._render(post_list, face_list, rhand_list, lhand_list)
            canvas = torch.FloatTensor(canvas) # [h, w, c]
            canvas = canvas.permute(2, 0, 1).contiguous().unsqueeze(0)
            
            ori_vis.append(canvas) # [1, c, h, w]
        ori_vis = torch.cat(ori_vis, dim=0)
        print(ori_vis.shape)
        ori_vis = torchvision.utils.make_grid(ori_vis, )
        self.logger.experiment.add_image("ori_vis", ori_vis, self.global_step)
        return loss
    
    def _tensor2numpy(self, points, x_anchor, y_anchor):
        """points: [V, 2]
        """
        v, c = points.size()

        points = torch.clamp(points, -1., 1.)
        points = torch.cat([points[:, 0:1] * 1280 + x_anchor, points[:, :, 1:2] * 720 + y_anchor, torch.ones((t, v, 1)).to(points.device)], dim=-1)  # [T, V, 2]
        points = points.view(3*v) # [T, 3V]
        points = points.detach().cpu().numpy() # [T, 3V]
        return points


    def _render(self, posepts, facepts, r_handpts, l_handpts):
        myshape = (720, 1280, 3)
        numkeypoints = 8
        canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
        canvas = renderface_sparse(facepts, canvas, numkeypoints, disp=False)
        canvas = renderhand(r_handpts, canvas)
        canvas = renderhand(l_handpts, canvas)[:, 280:1000, :] # [720, 720, 3]
        canvas = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC) # [256, 256, 3]

        # canvas = np.transpose(canvas, (2, 0, 1))
        # print(canvas.shape, type(canvas))
        # cv2.imwrite("hah.jpg", np.transpose(canvas, (2, 0, 1)))
        # print("done!!!")
        # exit()
        # canvas = Image.fromarray(canvas[:, :, [2,1,0]])

        # size = (455, 256) # int(256*1280/720)

        
        # canvas = canvas.resize((int(256*1280/720), 256), Image.ANTIALIAS) # []
        # print(canvas.size)
        # canvas = np.asarray(canvas)[(455-256)//2: (455-256)//2+256, :]
        # canvas = Image.fromarray(canvas)
        # canvas.save("test2" + '.png')
        # exit()
        return canvas # [3, 720, 1280]

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss


    def validation_step(self, batch, batch_idx):
        self.forward(batch, "val")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=2048)
        parser.add_argument('--n_hiddens', type=int, default=240)
        parser.add_argument('--n_res_layers', type=int, default=4)
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
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
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
        temporal_kernel_size = 4
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            gcn_transtcn_block(256, 256, kernel_size, 1, residual=False, **kwargs0),
            gcn_transtcn_block(256, 256, kernel_size, 1, **kwargs),
            gcn_transtcn_block(256, 128, kernel_size, 1, **kwargs),
            gcn_transtcn_block(128, 128, kernel_size, 1, **kwargs),
            gcn_transtcn_block(128, 128, kernel_size, 2, **kwargs),
            gcn_transtcn_block(128, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 64, kernel_size, 1, **kwargs),
            gcn_transtcn_block(64, 64, kernel_size, 2, **kwargs),
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

        return x


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

        return self.relu(x), A


class gcn_transtcn_block(nn.Module):
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
                 residual=True,
                 act_fn=nn.ReLU(inplace=True)):
        super().__init__()

        assert len(kernel_size) == 2

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SamePadConvTranspose2d(out_channels, out_channels, kernel_size, (stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                SamePadConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=kernel_size,
                                       stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        # self.relu = nn.ReLU(inplace=True)
        self.act = act_fn

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.act(x), A


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
        return self.convt(F.pad(x, self.pad_input))





if __name__ == "__main__":
    # N, C, T, V = 5, 2, 16, 25
    # x = torch.randn(N, C, T, V)

    # model = ST_GCN_18(2, 10, {"layout": 'ntu-rgb+d', "strategy": 'spatial'}) 
    # print(model)

    # out = model.extract_feature(x)
    # print(out.shape)
    x = torch.randn(2, 64, 4, 25)

    m1 = SamePadConv2d(64, 64, 4, (1,2))
    x = m1(x)
    print("x: ", x.shape)

    m2 = SamePadConvTranspose2d(64, 64, 5, (1,2))
    x = m2(x)

    print("x: ", x.shape)

    graph = Graph(layout= "pose25", strategy="spatial")
    A = torch.tensor(graph.A,
                    dtype=torch.float32,
                    requires_grad=False)

    # build networks
    spatial_kernel_size = A.size(0)
    print(spatial_kernel_size, A.size())
    kernel_size = (9, spatial_kernel_size)
    st_gcn = st_gcn_block(64, 64, kernel_size, 2)

    x, A = st_gcn(x, A)
    print("stgcn: ", x.shape)

    gcn_transtcn = gcn_transtcn_block(64, 64, (4, 3), 2)
    x, A = gcn_transtcn(x, A)
    print("trans tcn: ", x.shape)