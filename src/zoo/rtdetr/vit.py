import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, Height, Width, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.gelu = nn.GELU()
        self.convs = nn.ModuleList([])
        self.batchnorms = nn.ModuleList([])
        self.height = Height
        self.width = Width

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

            self.convs.append(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim))
            self.batchnorms.append(nn.BatchNorm2d(dim))

    def forward(self, x):
        for i, [attn, ff] in enumerate(self.layers):
            shortcut = x[:, 1:]
            shortcut = rearrange(shortcut, 'b (h w) d -> b d h w', h=self.height, w=self.width)
            shortcut = self.gelu(shortcut)
            shortcut = self.batchnorms[i](shortcut)
            shortcut = self.convs[i](shortcut)
            shortcut = rearrange(shortcut, 'b d h w -> b (h w) d')
            cls_tokens = torch.zeros(shortcut.shape[0], 1, shortcut.shape[2], device=shortcut.device)
            shortcut = torch.cat((cls_tokens, shortcut), dim=1)
            x = attn(x) + x
            x = ff(x) + x
            x = shortcut + x

        return x
