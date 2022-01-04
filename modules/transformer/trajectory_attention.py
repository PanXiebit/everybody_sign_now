#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import math


class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q_, k_, v_ = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print("q_, k_, v_: ", q_.shape, k_.shape, v_.shape) # [bs*heads, seq_len*frames, head_dim]

        # # remove CLS token from q, k, v
        # (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
        #     lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # # let CLS token attend to key / values of all patches across time and space
        # cls_out = qkv_attn(cls_q * self.scale, k, v)
        # cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)
        
        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
            space_attn = (self.scale * q_dot_k).softmax(dim=-1)
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # print("after orthoformer: ", x.shape)  # [bs*head, num_frame*path_num, num_frame, head_dim]
        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)  # [bs, num_frame*path_num, num_frame, dim]
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # [bs, num_frame, path_num, num_frame, dim]
        # print("x_diag: ", x_diag.shape)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)  # [bs, num_frame*path_num, dim]
        # print("x_diag: ", x_diag.shape)

        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)  # [bs, head, num_frame*path_num, head_dim]
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2)) # [bs, head, num_frame*path_num, head_dim]
        # print("q2, k2, v2: ", q2.shape, k2.shape, v2.shape)
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        # print("attn: ", attn.shape)    # [bs, head, num_frame*path_num, num_frame], 复杂度只有 path_num*num_frame**2

        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2) # [bs, head, num_frame*path_num, head_dim]
        x = rearrange(x, f'b h s d -> b s (h d)') # [bs, num_frame*path_num, dim]

        # concat back the cls token
        # x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def orthogonal_landmarks(q, k, num_landmarks=64, subsample_fraction=1.0):
    """
    Construct set of landmarks by recursively selecting new landmarks 
    that are maximally orthogonal to the existing set.
    Returns near orthogonal landmarks with shape (B, M, D).
    """
    if subsample_fraction < 1.0:
        # Need at least M/2 samples of queries and keys
        num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
        q_unnormalised = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :] # (B, N, D)
    else:
        # (B, N, D)
        q_unnormalised = q

    # may need to change default eps to eps=1e-8 for mixed precision compatibility
    qk = Fn.normalize(q_unnormalised, p=2, dim=-1)
    # print("qk: ", qk.shape)
    B, N, D = qk.shape

    selected_mask = torch.zeros((B, N, 1), device=qk.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

    # Get initial random landmark
    random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device) # int random from [0, N-1]
    # print("random_idx: ", random_idx.shape) # [bs, 1, 1]

    selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D) # 在第二个时间维度，随机选择一个 feature [D]. -> [B, D]
    # print("selected_landmark: ", selected_landmark.shape)  # [bs, haed_dim]

    selected_mask.scatter_(-2, random_idx, landmark_mask)  # [B, N, 1] 时间维度对应的idx转换为1， 其余仍为 0
    # print("selected_mask: ", selected_mask.shape)
    # Selected landmarks
    selected_landmarks = torch.empty((B, num_landmarks, D), device=qk.device, dtype=qk.dtype)
    selected_landmarks[:, 0, :] = selected_landmark    # [B, N, D] 第一个landmarks设置为 从qk随机选出来 selected_landmark, [B, D]
    # print("selected_landmarks: ", selected_landmarks.shape)

    # Store computed cosine similarities
    cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

    for M in range(1, num_landmarks):
        # Calculate absolute cosine similarity between selected and unselected landmarks
        # (B, N, D) * (B, D) -> (B, N)  # 求qk[B, N, D]和从qk每个时间维度随机选择一个的[B, D]的相似度，等价于矩阵相乘
        cos_sim = torch.einsum('b n d, b d -> b n', qk, selected_landmark).abs()  # [B, N]
        cos_sims[:, :, M - 1] = cos_sim  # 依次放在 M-1 的位置
        # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
        cos_sim_set = cos_sims[:, :, :M]  # 已经计算好的相似度集合

        # Get orthogonal landmark: landmark with smallest absolute cosine similarity:
        # set cosine similarity for already selected landmarks to > 1
        cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
        # (B,) - want max for non
        selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
        selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(B, D)

        # Add most orthogonal landmark to selected landmarks: 
        selected_landmarks[:, M, :] = selected_landmark

        # Removed selected indices from non-selected mask: 
        selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)
        
    landmarks = torch.masked_select(q_unnormalised, selected_mask.bool()).reshape(B, -1, D) # (B, M, D)
    return landmarks # (B, M, D)


def orthoformer(
    q, k, v, num_landmarks=64, subsample_fraction=1.0, 
    num_frames=None, shared_landmarks=True, return_attn=False
):
    """
    Computes spatial attention for all pairs of frames.
    The attention matrix is approximated using 
    intermediate landmarks taken from the queries and keys.
    The landmarks can be unique (to each frame) or 
    shared (a common set of landmarks across frames).

    q,k,v: [b*h, num_frame*path_num, head_dim]
    """
    B, N, D = k.shape
    F = num_frames
    L = num_landmarks
    P = N // F   # seq_len

    scale = D ** -0.25
    q = q * scale
    k = k * scale
    
    if shared_landmarks:
        with torch.no_grad():
            landmarks = orthogonal_landmarks(q, k, num_landmarks, subsample_fraction)
        # print("landmarks: ", landmarks.shape) # [bs, num_landmarks, head_dim]

        kernel_1 = Fn.softmax(torch.matmul(q, landmarks.transpose(-1, -2)), dim=-1) # [bs, num_frame*path_num, num_landmarks]
        # print("kernel_1: ", kernel_1.shape)
        
        kernel_2 = Fn.softmax(
            rearrange(torch.matmul(
                landmarks, k.transpose(-1, -2)), 'b l (f p) -> b l f p', f=F), dim=-1)
        # print("kernel_2: ", kernel_2.shape) # [bs, num_landmarks, num_frame, path_num], [b, l, f, p]

        
        v = rearrange(v, 'b (f p) d -> b f p d', f=F)  # [bs, num_frame, path_num, head_dim], [b, f, p, d]
        x = torch.einsum('b l f p, b f p d -> b l f d', kernel_2, v) # [bs, num_landmarks, num_frame, head_dim]
        x = torch.einsum('b n l, b l f d -> b n f d', kernel_1, x)  # [bs, num_frame*path_num, num_frame, head_dim]
        if return_attn:
            attn = torch.einsum('b m l, b l f p -> b m f p', kernel_1, kernel_2)
            return x, attn
    else:
        q = rearrange(q, 'b (f p) d -> (b f) p d', f=F)
        k = rearrange(k, 'b (g q) d -> (b g) q d', g=F)
        with torch.no_grad():
            landmarks = orthogonal_landmarks(q, k, num_landmarks, subsample_fraction)
            landmarks = rearrange(landmarks, '(b f) l d -> b f l d', f=F)
        q = rearrange(q, '(b f) p d -> b f 1 p d', f=F)
        k = rearrange(k, '(b g) q d -> b 1 g q d', g=F)
        v = rearrange(v, 'b (g q) d -> b 1 g q d', g=F)
        kernel_1 = Fn.softmax(
            torch.matmul(q, landmarks.unsqueeze(-4).transpose(-1, -2)), dim=-1)
        kernel_2 = Fn.softmax(
            torch.matmul(landmarks.unsqueeze(-3), k.transpose(-1, -2)), dim=-1)
        x = torch.matmul(kernel_1, torch.matmul(kernel_2, v))
        x = rearrange(x, 'b f g p d -> b (f p) g d')
        if return_attn:
            attn = torch.matmul(kernel_1, kernel_2)
            attn = rearrange(attn, 'b f g p q -> b (f p) g q')
            return x, attn

    return x


if __name__ == '__main__':
    attn = TrajectoryAttention(dim=256, num_heads=8, qkv_bias=False, 
        attn_drop=0., proj_drop=0., use_original_code=False)

    x = torch.randn(2, 196*8, 256)
    # print("input: ", x.shape)
    out = attn(x, seq_len=196, num_frames=8, approx="orthoformer")
    # print(out[0].shape)