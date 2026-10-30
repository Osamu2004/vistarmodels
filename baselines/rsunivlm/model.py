"""One official RSUniVLM checkpoint shared by both dense prediction tasks."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import MethodType
from typing import Any


def keep_pixel_expert_during_generation(model: Any) -> None:
    """Retain granularity=2 even when HF prepare_inputs drops extra kwargs.

    The upstream override calls Qwen2's preparation method but does not
    explicitly reinsert granularity. Enforce the pixel expert on every
    autoregressive step without changing third-party source or weights.
    """
    original = model.prepare_inputs_for_generation

    def prepare(self: Any, input_ids: Any, past_key_values: Any = None,
                inputs_embeds: Any = None, **kwargs: Any) -> dict[str, Any]:
        result = original(input_ids, past_key_values=past_key_values,
                          inputs_embeds=inputs_embeds, **kwargs)
        result["granularity"] = 2
        return result

    model.prepare_inputs_for_generation = MethodType(prepare, model)


class RSUniVLM:
    def __init__(self, source_root: Path, model_path: Path, *, device: str = "cuda:0",
                 dtype: str = "float16", attention: str = "eager",
                 max_new_tokens: int = 8192, vision_tower: str = "") -> None:
        import torch

        sys.path.insert(0, str(source_root))
        import llava
        if not Path(llava.__file__).resolve().is_relative_to(source_root.resolve()):
            raise RuntimeError("Another llava package is already imported; use a separate RSUniVLM environment")
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This official RSUniVLM adapter requires a CUDA device")
        torch.cuda.set_device(self.device)
        self.dtype = getattr(torch, dtype)
        kwargs: dict[str, Any] = {}
        # Upstream builder applies overwrite_config only in other model
        # branches. Supply an actual customized config for the GMoE branch.
        if vision_tower:
            from llava.model.language_model.llava_qwen_gmoe import LlavaQwenGMoeConfig
            config = LlavaQwenGMoeConfig.from_pretrained(str(model_path))
            config.mm_vision_tower = vision_tower
            kwargs = {"customized_config": config}
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            str(model_path), None, "llava_qwen_gmoe", multimodal=True,
            torch_dtype=dtype, attn_implementation=attention,
            device_map=str(self.device), **kwargs,
        )
        self.model.eval()
        # The official builder hardcodes float16 for the vision tower.
        self.model.get_vision_tower().to(device=self.device, dtype=self.dtype)
        keep_pixel_expert_during_generation(self.model)
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.template = conv_templates["qwen_1_5"]
        self.max_new_tokens = max_new_tokens

    def generate(self, images: list[Any], instruction: str) -> str:
        if len(images) not in (1, 2) or not instruction.startswith("[SEG]"):
            raise ValueError("Dense inference requires one/two images and a [SEG] instruction")
        question = " ".join(["<image>"] * len(images)) + "\n" + instruction
        conversation = copy.deepcopy(self.template)
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        tokens = self.tokenizer_image_token(
            conversation.get_prompt(), self.tokenizer, self.image_token_index, return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        if tokens.shape[1] >= self.context_len:
            raise ValueError("Prompt exceeds the checkpoint context; refusing to truncate image tokens")
        pixels = self.process_images(images, self.processor, self.model.config)
        # Match the official demo: one tensor per temporal image, whether
        # process_images returns a stack or an any-resolution tensor list.
        pixels = [x.to(device=self.device, dtype=self.dtype) for x in pixels]
        with self.torch.inference_mode():
            output = self.model.generate(
                tokens, images=pixels, image_sizes=[image.size for image in images],
                modalities=["image"] * len(images), granularity=2,
                do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
            )
        # The official GMoE generate uses inputs_embeds and returns new tokens;
        # it does not prepend input_ids. Do not slice by prompt length.
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
