from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedMaskBatchResult:
    images: torch.Tensor
    ratios: torch.Tensor


def _encode_prompts(ctx, prompts: Sequence[str]) -> torch.Tensor:
    cache = getattr(ctx, "_heterogeneous_prompt_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_heterogeneous_prompt_cache", cache)
    missing = list(dict.fromkeys(prompt for prompt in prompts if prompt not in cache))
    if missing:
        text_inputs = ctx.tokenizer(
            missing,
            padding="max_length",
            truncation=True,
            max_length=ctx.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            missing_embeds = ctx.text_encoder(text_inputs.input_ids.to(ctx.device))[0]
        missing_embeds = missing_embeds.to(dtype=ctx.weight_dtype)
        for index, prompt in enumerate(missing):
            cache[prompt] = missing_embeds[index : index + 1]
    return torch.cat([cache[prompt] for prompt in prompts], dim=0)


def _seeded_latents(ctx, seeds: Sequence[int], image_size: int) -> torch.Tensor:
    shape = (1, ctx.unet.config.in_channels, image_size // 8, image_size // 8)
    latents = []
    for seed in seeds:
        generator = torch.Generator(device=ctx.device).manual_seed(int(seed))
        latents.append(
            torch.randn(
                shape,
                generator=generator,
                device=ctx.device,
                dtype=ctx.weight_dtype,
            )
        )
    return torch.cat(latents, dim=0) * ctx.scheduler.init_noise_sigma


def generate_fixed_mask_batch(
    ctx,
    upstream,
    *,
    raw_masks: Sequence[torch.Tensor],
    prompts: Sequence[str],
    seeds: Sequence[int],
    image_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_rescale: float,
    control_scale: float,
    ignore_index: int = 255,
) -> FixedMaskBatchResult:
    """Generate one image per distinct fixed LoveDA mask in a true batch.

    The official ``generate_with_context`` repeats one mask, prompt, domain, and
    seed across ``ctx.batch_size``. This adapter preserves the released model
    computation while batching heterogeneous fixed masks, prompts, and seeds.
    """

    batch_size = int(ctx.batch_size)
    if not (len(raw_masks) == len(prompts) == len(seeds) == batch_size):
        raise ValueError(
            "raw_masks, prompts, and seeds must each match the resident "
            f"context batch_size={batch_size}"
        )

    layout_onehot = []
    for mask in raw_masks:
        if mask.ndim != 2:
            raise ValueError(f"expected a 2D raw LoveDA mask, got shape={tuple(mask.shape)}")
        onehot = upstream._onehot_from_mask(
            mask,
            ctx.num_classes,
            ignore_index,
            "loveda_raw",
        )
        layout_onehot.append(onehot)
    layout_onehot_small = torch.stack(layout_onehot, dim=0).to(
        device=ctx.device,
        dtype=ctx.weight_dtype,
    )
    layout_cond = F.interpolate(
        layout_onehot_small,
        size=(image_size, image_size),
        mode="nearest",
    )
    ratios = layout_onehot_small.mean(dim=(2, 3)).detach()

    prompt_embeds = _encode_prompts(ctx, prompts)
    uncond_embeds = ctx.uncond_embeds
    if uncond_embeds is None or int(uncond_embeds.shape[0]) != batch_size:
        raise ValueError("resident SyntheticGen context has incompatible unconditional embeddings")

    ratio_embeds = ctx.ratio_projector(
        ratios.to(device=ctx.device, dtype=ctx.weight_dtype)
    ).to(dtype=ctx.weight_dtype)
    layout_uncond = torch.zeros_like(layout_cond)
    ratio_uncond = torch.zeros_like(ratio_embeds)

    ctx.scheduler.set_timesteps(int(num_inference_steps), device=ctx.device)
    timesteps = ctx.scheduler.timesteps
    latents = _seeded_latents(ctx, seeds, int(image_size))
    do_cfg = float(guidance_scale) != 1.0

    with torch.no_grad():
        for timestep in timesteps:
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                encoder_hidden_states = torch.cat([uncond_embeds, prompt_embeds], dim=0)
                controlnet_cond = torch.cat([layout_uncond, layout_cond], dim=0)
                class_labels = torch.cat([ratio_uncond, ratio_embeds], dim=0)
            else:
                latent_model_input = latents
                encoder_hidden_states = prompt_embeds
                controlnet_cond = layout_cond
                class_labels = ratio_embeds

            if hasattr(ctx.scheduler, "scale_model_input"):
                latent_model_input = ctx.scheduler.scale_model_input(latent_model_input, timestep)

            down_samples, mid_sample = ctx.controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                class_labels=class_labels,
                return_dict=False,
            )
            down_samples, mid_sample = ctx.film_gate(class_labels, down_samples, mid_sample)
            if float(control_scale) != 1.0:
                down_samples = tuple(sample * float(control_scale) for sample in down_samples)
                mid_sample = mid_sample * float(control_scale)

            noise_pred = ctx.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
            ).sample

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)
                if float(guidance_rescale) > 0:
                    noise_pred = upstream.rescale_noise_cfg(
                        noise_pred,
                        noise_text,
                        guidance_rescale=float(guidance_rescale),
                    )
            latents = ctx.scheduler.step(noise_pred, timestep, latents).prev_sample

    images = upstream._vae_decode(ctx.vae, latents / ctx.vae.config.scaling_factor)
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if int(images.shape[0]) != batch_size:
        raise RuntimeError(
            f"SyntheticGen decoded {int(images.shape[0])} images for batch_size={batch_size}"
        )
    return FixedMaskBatchResult(images=images, ratios=ratios)
