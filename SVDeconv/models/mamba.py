import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING
from sacred import Experiment

from config_diffusercam import initialise
from utils.tupperware import tupperware

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("Mamba")
ex = initialise(ex)


def group_norm(num_channels: int, args: "tupperware"):
    return nn.GroupNorm(num_groups=args.num_groups, num_channels=num_channels)


class MambaBlock(nn.Module):
    """A simplified Mamba-inspired residual block."""

    def __init__(self, channels: int, args: "tupperware"):
        super().__init__()
        expansion = 2
        hidden = channels * expansion

        self.project_in = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            group_norm(hidden, args),
            nn.GELU(),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            group_norm(hidden, args),
            nn.GELU(),
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            group_norm(channels, args),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.project_in(x)
        x = self.depthwise(x)
        x = self.project_out(x)
        return x + residual


class Mamba(nn.Module):
    """A lightweight restoration backbone inspired by Mamba."""

    def __init__(self, args: "tupperware", in_c: int = 4):
        super().__init__()
        self.args = args
        self.pixelshuffle_ratio = args.pixelshuffle_ratio

        self.in_channels = in_c * args.pixelshuffle_ratio ** 2
        self.out_channels = 3 * args.pixelshuffle_ratio ** 2
        base_channels = 64

        self.entry = nn.Sequential(
            nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            group_norm(base_channels, args),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[MambaBlock(base_channels, args) for _ in range(8)]
        )

        self.exit = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            group_norm(base_channels, args),
            nn.GELU(),
            nn.Conv2d(base_channels, self.out_channels, kernel_size=1, bias=True),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.entry(img)
        x = self.blocks(x)
        x = self.exit(x)
        return x


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    model = Mamba(args, in_c=4).to(args.device)
    dummy = torch.rand(1, 4 * args.pixelshuffle_ratio ** 2, args.image_height // args.pixelshuffle_ratio, args.image_width // args.pixelshuffle_ratio).to(args.device)
    out = model(dummy)
    print("Mamba output shape:", out.shape)
