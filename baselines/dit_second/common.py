from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SecondManifestDataset(Dataset):
    def __init__(self, manifest: str):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def load(path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        return TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return self.load(row["target_image"]), self.load(row["source_image"]), self.load(row["target_mask_rgb"]), row


class ConditionedDiT(nn.Module):
    """Official DiT-B/2 with source and mask latents concatenated at patch input."""

    def __init__(self, dit_root: str):
        super().__init__()
        sys.path.insert(0, str(Path(dit_root).expanduser().resolve()))
        from models import DiT_models
        self.model = DiT_models["DiT-B/2"](input_size=32, num_classes=1)
        old = self.model.x_embedder.proj
        self.model.x_embedder.proj = nn.Conv2d(12, old.out_channels, kernel_size=2, stride=2, bias=True)
        nn.init.xavier_uniform_(self.model.x_embedder.proj.weight.view(old.out_channels, -1))
        nn.init.zeros_(self.model.x_embedder.proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, condition: torch.Tensor):
        return self.model(torch.cat([x, condition], dim=1), t, y)


@torch.no_grad()
def encode(vae, image: torch.Tensor) -> torch.Tensor:
    return vae.encode(image).latent_dist.sample() * 0.18215


@torch.no_grad()
def decode(vae, latent: torch.Tensor) -> torch.Tensor:
    image = vae.decode(latent / 0.18215).sample
    return (image.clamp(-1, 1) + 1) / 2
