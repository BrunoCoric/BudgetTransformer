from timm.models.vision_transformer import Block, VisionTransformer, Attention
from torch import nn
import torch
import torch.nn.functional as F


class NystromAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_landmarks=64):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_landmarks = num_landmarks
        self.head_dim = dim // num_heads

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
        V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q /= self.scale

        segs = N // self.num_landmarks
        if N % self.num_landmarks == 0:
            Q_landmarks = q.reshape(-1, self.num_heads, self.num_landmarks, segs,
                                    self.head_dim).mean(dim=-2)
            K_landmarks = k.reshape(-1, self.num_heads, self.num_landmarks, segs,
                                    self.head_dim).mean(dim=-2)
        else:
            num_k = (segs + 1) * self.num_landmarks - N
            keys_landmarks_f = k[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(
                dim=-2)
            keys_landmarks_l = k[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.num_landmarks - num_k,
                                                                 segs + 1,
                                                                 self.head_dim).mean(dim=-2)
            K_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim=-2)

            queries_landmarks_f = q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(
                dim=-2)
            queries_landmarks_l = q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.num_landmarks - num_k,
                                                                    segs + 1,
                                                                    self.head_dim).mean(dim=-2)
            Q_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim=-2)

        kernel_1 = F.softmax(torch.matmul(q, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel_3 = F.softmax(torch.matmul(Q_landmarks, k.transpose(-1, -2)), dim=-1)
        x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NystromBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_landmarks=64):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path,
                         act_layer, norm_layer)
        self.attn = NystromAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, num_landmarks=num_landmarks)


class Nystromformer(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_landmarks=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__(img_size=img_size, num_classes=num_classes, embed_dim=embed_dim, norm_layer=norm_layer,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, patch_size=patch_size, in_chans=in_chans)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            NystromBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, num_landmarks=num_landmarks)
            for i in range(depth)])
