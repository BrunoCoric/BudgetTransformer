from timm.models.vision_transformer import Block, VisionTransformer, Attention
from torch import nn
import torch
import torch.nn.functional as F

def apply_rotary_pos_emb(q, k, sinusoidal_pos):
    cos_pos = torch.repeat_interleave(sinusoidal_pos[:, None, :, 1::2], 2, -1)
    sin_pos = torch.repeat_interleave(sinusoidal_pos[:, None, :, 0::2], 2, -1)
    q2 = torch.stack([-q[..., 1::2], q[..., 0::2]], -1)
    q2 = torch.reshape(q2, q.shape)
    q = q*cos_pos + q2*sin_pos
    k2 = torch.stack([-k[..., 1::2], k[..., 0::2]], -1)
    k2 = torch.reshape(k2, k.shape)
    k = k * cos_pos + k2 * sin_pos
    return q,k

class RotaryAttention(Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, rotary_embedding):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, seq_length, dim//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q, k = apply_rotary_pos_emb(q, k, rotary_embedding)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x