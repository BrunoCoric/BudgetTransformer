import time
import torch
from timm.models.vision_transformer import VisionTransformer
from nystrom_transformer import Nystromformer
from functools import partial
from models import DistilledNystromformer, DistilledVisionTransformer
from torch import nn


nyst = DistilledNystromformer(img_size=384,patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_landmarks=64)
vis = DistilledVisionTransformer(img_size=384,patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
# checkpoint = torch.hub.load_state_dict_from_url(
#                "https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth", map_location='cpu', check_hash=True)
#
# checkpoint_model = checkpoint['model']
# state_dict = model.state_dict()
# for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]
#
# # interpolate position embedding
# pos_embed_checkpoint = checkpoint_model['pos_embed']
# embedding_size = pos_embed_checkpoint.shape[-1]
# num_patches = model.patch_embed.num_patches
# num_extra_tokens = model.pos_embed.shape[-2] - num_patches
# # height (== width) for the checkpoint position embedding
# orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
# # height (== width) for the new position embedding
# new_size = int(num_patches ** 0.5)
# # class_token and dist_token are kept unchanged
# extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
# # only the position tokens are interpolated
# pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
# pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
# pos_tokens = torch.nn.functional.interpolate(
# pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
# pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
# new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
# checkpoint_model['pos_embed'] = new_pos_embed
# torch.set_flush_denormal(True)
# model.load_state_dict(checkpoint_model, strict=False)
t = torch.rand(10, 3, 384, 384)

start = time.time()
nyst(t)
print("Nystromformer", time.time()-start)
start = time.time()
vis(t)
print("Vision", time.time()-start)
