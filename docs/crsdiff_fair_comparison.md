# CRS-Diff Fair Comparison Protocol

CRS-Diff is useful as an external remote-sensing controllable generation
baseline, but it is not a task-identical competitor to UniGen.

## Fair Direct Comparison

Use CRS-Diff for condition-to-image generation:

```text
rendered semantic/change mask -> RGB remote-sensing image
```

Protocol:

1. Use the same eval split as UniGen.
2. Use the same resized GT RGB images used by UniGen generation eval.
3. Use the same rendered condition mask images saved by UniGen eval.
4. Feed the condition mask to CRS-Diff's `seg` local-control slot.
5. Generate at CRS-Diff native resolution, then resize generated images back to
   UniGen eval size before computing metrics.
6. Compute metrics from the same `gt_rgb` and `pred_rgb` directory structure:
   PSNR/SSIM/LPIPS, FID/FDr6 components, DINOv3-Sat FID, and CMMD.

This compares generation quality under the same condition masks and target
images.

## Not Fully Fair Without Extra Work

For SECOND change generation, UniGen conditions on:

```text
pre_image + post_class_change_mask -> post_image
post_image + pre_class_change_mask -> pre_image
```

Released CRS-Diff does not natively condition on both a source image and a
semantic change mask in the same way. Feeding only the change mask to CRS-Diff is
valid as a mask-conditioned generation baseline, but it is weaker and should be
reported as:

```text
CRS-Diff (change-mask only)
```

If source-image conditioning is added through CRS-Diff's content/global pathway,
report it separately as:

```text
CRS-Diff (change-mask + source-content)
```

and state that the source image is represented by a global content embedding,
not by pixel-aligned spatial conditioning.

## Recommended Tables

Ordinary generation:

```text
Pix2PixHD
SPADE/OASIS
ControlNet-Seg or T2I-Adapter-Seg
CRS-Diff (semantic-mask)
Flux LoRA gen-only
UniGen unified
```

Change generation:

```text
Copy source image
SD/Flux inpainting with change mask
InstructPix2Pix
CRS-Diff (change-mask only)
Flux LoRA change-gen-only
UniGen unified
```

The most important internal comparisons are still:

```text
gen-only Flux LoRA
change-gen-only Flux LoRA
unified four-task UniGen
```

These share the same backbone/data/task definition and are therefore stronger
evidence than external zero-shot baselines.

