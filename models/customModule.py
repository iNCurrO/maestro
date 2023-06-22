import torch
import numpy as np

class PatchEmbed(torch.nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            num_det: int = 724,
            num_views: int = 720,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_det = num_det

        # Input size (B, 1, D, V)
        self.proj = torch.nn.Conv2d(1, embed_dim, kernel_size=(1, self.num_det), stride=(1, 1), bias=bias)
        # Output Size (B, embed_dim, 1, V)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        B, C, D, V = x.shape
        if C==3:
            x = torch.mean(x, dim=1)
        assert(x.shape[1] == 1, f"Input image channel is wrong ({C}), please check it")
        assert(D == self.num_det, f"Input image height ({D}) doesn't match model ({self.num_det}).")
        assert(V == self.num_views, f"Input image width ({V}) doesn't match model ({self.num_views}).")
        # shape of x (B, 1, D ,V)
        x = self.proj(x)
        # shape of x (B, embed_dim, 1, V)
        x = x.flatten(2).transpose(1, 2)  # NC1V -> NLC
        x = self.norm(x)
        return x

def get_1d_sincos_pos_embed(embed_dim: int = 768, num_views: int = 720, cls_token=False):
    pos = np.arange(num_views, dtype=np.float32)
    assert embed_dim % 2 == 0, f"Embeded dimension ({embed_dim}) should be even, for the position encoding!\n"
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    out = np.einsum('m,d->md', pos, omega)
    
    pos_sin = np.sin(out)
    pos_cos = np.cos(out)
    
    pos_emb = np.concatenate([pos_sin, pos_cos], axis=1)
    if cls_token:
        pos_emb_cls = np.zeros([1, embed_dim])
    else:
        pos_emb_cls = None
    return pos_emb, pos_emb_cls
