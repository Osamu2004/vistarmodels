"""Runtime adapter for the official VIP/dino.txt implementation.

The pinned VIP release contains the model architecture and prompt templates,
but leaves the three DINOv3 asset paths empty.  This module instantiates the
same released architecture and loads those assets from explicit managed paths.
It intentionally has no MMSeg dependency; dataset handling and metrics live in
``eval_vip.py``.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from baselines.vip.protocols import (
    compute_square_padding,
    flatten_class_groups,
    normalize_path,
    sliding_window_boxes,
)


VIP_SOURCE_REVISION = "5bd25ee03ec25c1538622cf7da661e8c0461e769"
DINOV3_BACKBONE_HASH_PREFIX = "8aa4cbdd"
DINOTXT_HEAD_HASH_PREFIX = "a442d8f5"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class VIPLoadDiagnostics:
    """State-dict diagnostics retained in the evaluation metadata."""

    backbone_missing_keys: tuple[str, ...]
    backbone_unexpected_keys: tuple[str, ...]
    dinotxt_missing_keys: tuple[str, ...]
    dinotxt_unexpected_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "backbone_missing_keys": list(self.backbone_missing_keys),
            "backbone_unexpected_keys": list(self.backbone_unexpected_keys),
            "dinotxt_missing_keys": list(self.dinotxt_missing_keys),
            "dinotxt_unexpected_keys": list(self.dinotxt_unexpected_keys),
        }


def _load_torch_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "teacher"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested:
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Checkpoint is not a non-empty state dict: {path}")
    if not all(isinstance(key, str) for key in payload):
        raise TypeError(f"Checkpoint contains non-string keys: {path}")
    return payload


def _assert_source_import(source_root: Path) -> None:
    loaded = sys.modules.get("dinov3")
    if loaded is not None:
        module_file = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if source_root not in module_file.parents:
            raise RuntimeError(
                "A different dinov3 package was imported before VIP setup: "
                f"{module_file}. Start evaluation in a clean VIP environment."
            )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def build_official_vip_dinotxt(
    *,
    source_root: str | Path,
    backbone_checkpoint: str | Path,
    dinotxt_checkpoint: str | Path,
    bpe_vocabulary: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, VIPLoadDiagnostics]:
    """Build the released VIP dino.txt network with explicit local assets."""

    source_root = normalize_path(source_root)
    backbone_checkpoint = normalize_path(backbone_checkpoint)
    dinotxt_checkpoint = normalize_path(dinotxt_checkpoint)
    bpe_vocabulary = normalize_path(bpe_vocabulary)
    _assert_source_import(source_root)

    backbones = importlib.import_module("dinov3.hub.backbones")
    dinotxt_module = importlib.import_module("dinov3.eval.text.dinotxt_model")
    text_transformer_module = importlib.import_module(
        "dinov3.eval.text.text_transformer"
    )
    tokenizer_module = importlib.import_module("dinov3.eval.text.tokenizer")

    vision_backbone = backbones.dinov3_vitl16(pretrained=False)
    backbone_result = vision_backbone.load_state_dict(
        _load_torch_state_dict(backbone_checkpoint), strict=True
    )

    config = dinotxt_module.DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        # The VIP release explicitly requests all 24 intermediate layers and
        # consumes the final one.  Preserve that public implementation choice.
        vision_model_patch_token_layer=24,
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    text_backbone = text_transformer_module.TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = dinotxt_module.DINOTxt(
        model_config=config,
        vision_backbone=vision_backbone,
        text_backbone=text_backbone,
    )
    dinotxt_result = model.load_state_dict(
        _load_torch_state_dict(dinotxt_checkpoint), strict=False
    )
    unexpected = tuple(dinotxt_result.unexpected_keys)
    invalid_missing = tuple(
        key
        for key in dinotxt_result.missing_keys
        if not key.startswith("visual_model.backbone.")
    )
    if unexpected or invalid_missing:
        raise RuntimeError(
            "The managed dino.txt checkpoint is incompatible with the pinned "
            f"VIP architecture: unexpected={list(unexpected)[:8]}, "
            f"invalid_missing={list(invalid_missing)[:8]}"
        )

    tokenizer = tokenizer_module.Tokenizer(vocab_path=str(bpe_vocabulary))
    model.eval().to(device=device).half()
    diagnostics = VIPLoadDiagnostics(
        backbone_missing_keys=tuple(backbone_result.missing_keys),
        backbone_unexpected_keys=tuple(backbone_result.unexpected_keys),
        dinotxt_missing_keys=tuple(dinotxt_result.missing_keys),
        dinotxt_unexpected_keys=unexpected,
    )
    return model, tokenizer, diagnostics


class VIPSegmenter:
    """Inference-only implementation of the released VIP aggregation rule."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        backbone_checkpoint: str | Path,
        dinotxt_checkpoint: str | Path,
        bpe_vocabulary: str | Path,
        class_groups: Sequence[Sequence[str]],
        device: torch.device,
        logit_scale: float = 40.0,
        tau: float = 4.0,
        temperature: float = 1.0,
        probability_threshold: float = 0.0,
        background_index: int = 0,
        text_template: str = "openai_imagenet_template",
        crop_size: int = 336,
        stride: int = 112,
    ) -> None:
        if crop_size != 336:
            raise ValueError(
                "The released VIP vision head fixes a 21x21 patch grid, so "
                "crop_size must be 336 for a ViT-L/16 backbone."
            )
        if stride <= 0:
            raise ValueError("VIP stride must be positive")
        if tau <= 0 or temperature <= 0:
            raise ValueError("VIP tau and temperature must be positive")
        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError("VIP probability threshold must lie in [0,1]")

        self.device = device
        self.class_groups = tuple(tuple(group) for group in class_groups)
        aliases, indices = flatten_class_groups(self.class_groups)
        self.aliases = aliases
        self.num_classes = len(self.class_groups)
        self.query_indices = torch.tensor(
            indices, dtype=torch.long, device=device
        )
        self.logit_scale = float(logit_scale)
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.probability_threshold = float(probability_threshold)
        self.background_index = int(background_index)
        self.crop_size = int(crop_size)
        self.stride = int(stride)

        self.model, self.tokenizer, self.load_diagnostics = (
            build_official_vip_dinotxt(
                source_root=source_root,
                backbone_checkpoint=backbone_checkpoint,
                dinotxt_checkpoint=dinotxt_checkpoint,
                bpe_vocabulary=bpe_vocabulary,
                device=device,
            )
        )
        prompt_module = importlib.import_module("prompts.imagenet_template")
        templates = prompt_module.get_text_template(text_template)
        query_features: list[torch.Tensor] = []
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            for alias in self.aliases:
                prompts = [template(alias) for template in templates]
                tokens = self.tokenizer.tokenize(prompts).to(device)
                total = self.model.encode_text(tokens)
                # VIP uses the patch-aligned half of dino.txt's 2048-D output.
                feature = F.normalize(total[:, 1024:].clone(), dim=-1)
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

    @torch.inference_mode()
    def _forward_crop(self, crop: torch.Tensor) -> torch.Tensor:
        original_height, original_width = crop.shape[-2:]
        left, right, top, bottom = compute_square_padding(
            original_height, original_width, self.crop_size
        )
        if any((left, right, top, bottom)):
            crop = F.pad(crop, (left, right, top, bottom))

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, image_features = self.model.encode_image_with_patch_tokens(crop)
            image_features = F.normalize(image_features, dim=-1)
            patch_size = int(self.model.visual_model.backbone.patch_size)
            rows = crop.shape[-2] // patch_size
            columns = crop.shape[-1] // patch_size
            if rows * columns != image_features.shape[1]:
                raise RuntimeError(
                    "VIP patch-token count does not match the padded crop: "
                    f"grid={rows}x{columns}, tokens={image_features.shape[1]}"
                )

            alias_logits = torch.einsum(
                "bnd,qkd->bnqk", image_features, self.query_features
            ).mean(dim=-1)
            alias_logits = alias_logits.permute(0, 2, 1).reshape(
                crop.shape[0], len(self.aliases), rows, columns
            )

            average_image = F.normalize(image_features.mean(dim=1), dim=-1)
            average_text = F.normalize(self.query_features.mean(dim=1), dim=-1)
            class_scores = (average_image @ average_text.T).squeeze(0)
            alias_logits = alias_logits.squeeze(0) * self.logit_scale

            class_logits = torch.zeros(
                (self.num_classes, rows, columns),
                device=alias_logits.device,
                dtype=alias_logits.dtype,
            )
            for class_id in range(self.num_classes):
                selected = self.query_indices == class_id
                weights = torch.softmax(
                    class_scores[selected] / self.temperature, dim=0
                )
                weighted = (
                    alias_logits[selected]
                    * weights[:, None, None]
                    * (1.0 / weights.mean())
                )
                class_logits[class_id] = torch.logsumexp(
                    self.tau * weighted, dim=0
                ) / self.tau

            dense = F.interpolate(
                class_logits.unsqueeze(0),
                size=(crop.shape[-2], crop.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        if any((left, right, top, bottom)):
            dense = dense[
                :,
                :,
                top : top + original_height,
                left : left + original_width,
            ]
        return dense.float()

    @torch.inference_mode()
    def predict(
        self,
        image_rgb: np.ndarray,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Predict an already-resized RGB image and restore the output extent."""

        image = np.asarray(image_rgb, dtype=np.float32) / 255.0
        normalized = (image - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(normalized.transpose(2, 0, 1))
        ).unsqueeze(0).to(device=self.device, dtype=torch.float16)
        height, width = tensor.shape[-2:]
        logits = torch.zeros(
            (1, self.num_classes, height, width),
            device=self.device,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            (1, 1, height, width), device=self.device, dtype=torch.float32
        )
        boxes = sliding_window_boxes(
            height, width, self.crop_size, self.stride
        )
        for y1, y2, x1, x2 in boxes:
            crop_logits = self._forward_crop(tensor[:, :, y1:y2, x1:x2])
            logits[:, :, y1:y2, x1:x2] += crop_logits
            counts[:, :, y1:y2, x1:x2] += 1
        if bool(torch.any(counts == 0)):
            raise RuntimeError("VIP sliding windows left uncovered pixels")
        logits = logits / counts
        if output_size is not None and tuple(output_size) != (height, width):
            logits = F.interpolate(
                logits,
                size=tuple(int(value) for value in output_size),
                mode="bilinear",
                align_corners=False,
            )
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        prediction[confidence < self.probability_threshold] = self.background_index
        output = prediction.squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        return np.asarray(output, dtype=np.int64), len(boxes)
