from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


REQUIRED_MANIFEST_KEYS = (
    "name",
    "source_image",
    "target_image",
    "target_mask_ids",
    "target_mask_rgb",
)
SD_VAE_SCALE = 0.18215


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest: str | Path, verify_files: bool = True) -> list[dict[str, Any]]:
    path = resolve_path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"SECOND manifest does not exist: {path}")

    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        missing = [key for key in REQUIRED_MANIFEST_KEYS if not row.get(key)]
        if missing:
            raise ValueError(f"{path}:{line_number} is missing required fields: {missing}")
        name = str(row["name"])
        if name in names:
            raise ValueError(f"duplicate sample name in {path}:{line_number}: {name}")
        names.add(name)
        if verify_files:
            for key in REQUIRED_MANIFEST_KEYS[1:]:
                candidate = resolve_path(row[key])
                if not candidate.is_file():
                    raise FileNotFoundError(f"{path}:{line_number} field {key} does not exist: {candidate}")
        rows.append(row)
    if not rows:
        raise ValueError(f"SECOND manifest is empty: {path}")
    return rows


def load_rgb(path: str | Path, image_size: int, *, is_mask: bool = False) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        image = image.resize((image_size, image_size), resample)
    return TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)


class SecondManifestDataset(Dataset):
    """Directional SECOND tuples: source RGB + target change mask -> target RGB."""

    def __init__(
        self,
        manifest: str,
        image_size: int = 256,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        verify_files: bool = True,
    ):
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by the SD VAE factor 8, got {image_size}")
        for name, value in (("hflip_prob", hflip_prob), ("vflip_prob", vflip_prob)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.manifest = str(resolve_path(manifest))
        self.rows = read_manifest(manifest, verify_files=verify_files)
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target = load_rgb(row["target_image"], self.image_size)
        source = load_rgb(row["source_image"], self.image_size)
        mask = load_rgb(row["target_mask_rgb"], self.image_size, is_mask=True)

        # Apply geometry jointly so the source, condition, and target stay registered.
        if self.hflip_prob and random.random() < self.hflip_prob:
            target, source, mask = (TF.hflip(value) for value in (target, source, mask))
        if self.vflip_prob and random.random() < self.vflip_prob:
            target, source, mask = (TF.vflip(value) for value in (target, source, mask))

        # Do not return the full manifest row: changed_class_ids is variable-length
        # and PyTorch's default collator cannot stack it reliably.
        return {
            "target": target,
            "source": source,
            "mask": mask,
            "name": str(row["name"]),
            "direction": str(row.get("direction", "unknown")),
            "index": index,
        }


class ConditionedDiT(nn.Module):
    """Official DiT-B/2 with source and mask latents concatenated at patch input.

    The denoised target occupies four latent channels. The source RGB and
    directional semantic-change mask are independently encoded by the frozen
    SD VAE and provide eight additional spatial channels. DiT still predicts
    only the target epsilon/sigma channels.
    """

    target_channels = 4
    condition_channels = 8

    def __init__(self, dit_root: str, latent_size: int = 32):
        super().__init__()
        root = resolve_path(dit_root)
        if not (root / "models.py").is_file():
            raise FileNotFoundError(f"official DiT models.py is missing under: {root}")
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from models import DiT_models

        self.model = DiT_models["DiT-B/2"](
            input_size=latent_size,
            num_classes=1,
            class_dropout_prob=0.0,
        )
        old = self.model.x_embedder.proj
        input_channels = self.target_channels + self.condition_channels
        self.model.x_embedder.proj = nn.Conv2d(
            input_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        nn.init.xavier_uniform_(self.model.x_embedder.proj.weight.view(old.out_channels, -1))
        if self.model.x_embedder.proj.bias is not None:
            nn.init.zeros_(self.model.x_embedder.proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, condition: torch.Tensor):
        if x.shape[1] != self.target_channels:
            raise ValueError(f"target latent must have {self.target_channels} channels, got {x.shape}")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(f"condition latent must have {self.condition_channels} channels, got {condition.shape}")
        if x.shape[0] != condition.shape[0] or x.shape[-2:] != condition.shape[-2:]:
            raise ValueError(f"target/condition latent shapes are incompatible: {x.shape} vs {condition.shape}")
        return self.model(torch.cat([x, condition], dim=1), t, y)


@torch.no_grad()
def encode(vae: nn.Module, image: torch.Tensor, *, sample_posterior: bool = True) -> torch.Tensor:
    posterior = vae.encode(image).latent_dist
    latent = posterior.sample() if sample_posterior else posterior.mode()
    return latent * SD_VAE_SCALE


@torch.no_grad()
def decode(vae: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    image = vae.decode(latent / SD_VAE_SCALE).sample
    return (image.clamp(-1, 1) + 1) / 2


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
