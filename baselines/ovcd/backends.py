"""Pinned official DynamicEarth M-C-I and training-free OpenDPR inference.

No upstream files are edited. Each process imports exactly one upstream
dynamic_earth package; keep the two methods in separate processes/environments.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REVISIONS = dict(dynamicearth="c9ffd90cafbd791cd75a48a5717a902966c2436c",
                 opendpr="bdf12a3d3e5adccd4e2c166a0c3a5d9f5a19c313")
NAMES = ("water", "ground", "low vegetation", "tree", "building", "playground")


def union_masks(masks, shape):
    result = np.zeros(shape, np.uint8)
    for mask in masks:
        array = np.asarray(mask)
        if array.shape != shape:
            raise ValueError("Upstream proposal extent differs from the input")
        result[array.astype(bool)] = 1
    return result


def configure_source(root):
    root = Path(root).resolve()
    previous = sys.modules.get("dynamic_earth")
    if previous and not str(Path(previous.__file__).resolve()).startswith(str(root) + "/"):
        raise RuntimeError("Another dynamic_earth package is loaded; run each method in its own process")
    for path in reversed((root, root / "third_party/segment_anything", root / "third_party/SegEarth_OV",
                           root / "third_party/SimFeatUp")):
        if path.is_dir():
            sys.path.insert(0, str(path))


def sam_proposer(args, module):
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).to(args.device).eval()
    proposer = module.MaskProposal()
    return model, proposer


SAM_DEFAULTS = dict(points_per_side=32, points_per_batch=64, pred_iou_thresh=0.5,
    stability_score_thresh=0.95, stability_score_offset=0.9, box_nms_thresh=0.7, min_mask_region_area=0)


class OpenDPR:
    def __init__(self, args, dataset, output):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        configure_source(args.source_root)
        from dynamic_earth import mask_generator
        from dynamic_earth.comparator.bi_match import bitemporal_match
        from dynamic_earth.identifier.prototype_retrieval import build_prototype_set, identify
        self.args, self.dataset = args, dataset
        self.model, self.proposer = sam_proposer(args, mask_generator)
        self.proposer.make_mask_generator(model=self.model, **SAM_DEFAULTS)
        self.proposer.set_hyperparameters()
        self.encoder = AutoModel.from_pretrained(args.dinov2_dir, local_files_only=True).to(args.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(args.dinov2_dir, local_files_only=True,
                                                            do_resize=False, do_center_crop=False)
        self.config = dict(model_type="DINOv2", feature_dim=768, patch_size=14,
                           max_num_prototypes=10 if dataset == "second" else 5)
        classes = list(NAMES) if dataset == "second" else ["background", "building"]
        support = Path(args.source_root) / "support_set" / ("SECOND" if dataset == "second" else "LEVIR-CD")
        with torch.inference_mode():
            self.prototypes = build_prototype_set(self.encoder, self.processor, self.config, classes,
                                                  str(support), args.device)
        labels = set(self.prototypes["prototype_labels"].tolist())
        if labels != set(range(len(classes))):
            raise ValueError(f"Missing support prototypes: expected{classes}, got IDs{sorted(labels)}")
        self.match, self.identify = bitemporal_match, identify

    def predict(self, first, second):
        from segment_anything.utils.amg import rle_to_mask
        proposal, first_count = self.proposer.forward(first, second)
        masks = np.asarray([rle_to_mask(rle).astype(bool) for rle in proposal["rles"]])
        count = 6 if self.dataset == "second" else 1
        if not len(masks):
            return [np.zeros(first.shape[:2], np.uint8) for _ in range(count)]
        changed, first_count = self.match(first, second, masks, self.encoder, self.processor, first_count,
            model_config=self.config, device=self.args.device, change_confidence_threshold=145)
        if not len(changed):
            return [np.zeros(first.shape[:2], np.uint8) for _ in range(count)]
        result, _, _, _ = self.identify(first, second, changed, first_count, self.encoder, self.processor,
            prototype_state=self.prototypes, model_config=self.config, device=self.args.device,
            is_instance_class=self.dataset == "levircd", num_classes=6 if self.dataset == "second" else 2)
        if self.dataset == "levircd":
            return [union_masks(result, first.shape[:2])]
        # Official SECOND class IDs0..5 are mapped once to benchmark IDs1..6.
        if set(result) != set(range(6)):
            raise ValueError("OpenDPR did not return all six category memberships")
        return [union_masks(result[c], first.shape[:2]) for c in range(6)]


class DynamicEarth:
    def __init__(self, args, dataset, output):
        import torch
        from torchvision import transforms
        configure_source(args.source_root)
        from dynamic_earth import sam_ext
        from dynamic_earth.comparator.bi_match import bitemporal_match
        from dynamic_earth.identifier.utils import identify
        import dynamic_earth.identifier.segearth_ov_ext as segmentation
        self.args, self.dataset = args, dataset
        settings = json.loads(Path(args.dynamic_config).read_text())["classes"]
        self.classes = settings if dataset == "second" else [dict(class_id=1,
            background=["background"], foreground=["building"], sam=dict(SAM_DEFAULTS),
            change_threshold=145, is_instance_class=True)]
        self.sam, self.proposer = sam_proposer(args, sam_ext)
        self.comparator = torch.hub.load(args.dino_repo, "dino_vitb16", source="local", pretrained=False)
        state = torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
        self.comparator.load_state_dict(state, strict=True)
        self.comparator = self.comparator.to(args.device).eval()
        self.comparator_processor = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
        self.processor = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711)),
            transforms.Resize((448,448))])
        names = Path(output) / "class_names.txt"
        names.parent.mkdir(parents=True, exist_ok=True)
        names.write_text("background\nbuilding\n")
        # This published Lightning checkpoint predates Torch's weights_only
        # default. Scope the trusted-file compatibility to that exact path.
        from unittest.mock import patch
        original_load = torch.load
        def load_checkpoint(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == Path(args.segearth_weight).resolve():
                kwargs.setdefault("map_location", "cpu")
                kwargs.setdefault("weights_only", False)
            return original_load(path, *positional, **kwargs)
        with patch.object(torch, "load", load_checkpoint):
            self.identifier = segmentation.SegEarth_OV(clip_type="CLIP", vit_type="ViT-B/16",
                model_type="SegEarth", name_path=str(names), device=torch.device(args.device),
                ignore_residual=True, feature_up=True,
                feature_up_cfg=dict(model_name="jbu_one", model_path=args.segearth_weight), cls_token_lambda=-0.3).eval()
        self.text_states = []
        with torch.inference_mode():
            for entry in self.classes:
                words = entry["background"] + entry["foreground"]
                indices = [0] * len(entry["background"]) + [1] * len(entry["foreground"])
                features = []
                for word in words:
                    tokens = self.identifier.tokenizer([template(word) for template in
                        segmentation.openai_imagenet_template]).to(args.device)
                    feature = self.identifier.net.encode_text(tokens)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    features.append(feature.unsqueeze(0))
                self.text_states.append((torch.cat(features), torch.tensor(indices, device=args.device)))
        self.match, self.identify = bitemporal_match, identify

    def predict(self, first, second):
        from segment_anything.utils.amg import rle_to_mask
        proposals, comparisons, predictions = {}, {}, []
        for entry, (features, indices) in zip(self.classes, self.text_states):
            key = json.dumps(entry["sam"], sort_keys=True)
            if key not in proposals:
                self.proposer.make_mask_generator(model=self.sam, **entry["sam"])
                self.proposer.set_hyperparameters()
                raw, first_count = self.proposer.forward(first, second)
                masks = np.asarray([rle_to_mask(rle).astype(bool) for rle in raw["rles"]])
                proposals[key] = masks, first_count
            masks, first_count = proposals[key]
            if not len(masks):
                predictions.append(np.zeros(first.shape[:2], np.uint8))
                continue
            comparison_key = key, entry["change_threshold"]
            if comparison_key not in comparisons:
                comparisons[comparison_key] = self.match(first, second, masks, self.comparator,
                    self.comparator_processor, first_count, change_confidence_threshold=entry["change_threshold"],
                    device=self.args.device, model_config=dict(model_type="DINO",feature_dim=768,patch_size=16))
            changed, changed_first = comparisons[comparison_key]
            if not len(changed):
                predictions.append(np.zeros(first.shape[:2], np.uint8))
                continue
            model = self.identifier
            model.query_features, model.query_idx = features, indices
            model.num_queries, model.num_classes = len(indices), 2
            identified, _ = self.identify(first, second, changed, changed_first, model, self.processor,
                device=self.args.device, is_instance_class=entry["is_instance_class"])
            predictions.append(union_masks(identified, first.shape[:2]))
        return predictions
