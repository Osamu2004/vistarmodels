import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TR


class SecondDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 256 if opt.phase == "test" or for_metrics else 286
        opt.crop_size = 256
        opt.label_nc = 7
        opt.contain_dontcare_label = False
        opt.semantic_nc = 7
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0
        self.opt = opt
        self.for_metrics = for_metrics
        split = "test" if opt.phase == "test" or for_metrics else "train"
        root = os.path.join(opt.dataroot, split)
        self.image_dir = os.path.join(root, "target_rgb")
        self.label_dir = os.path.join(root, "target_mask_ids")
        self.names = sorted(name for name in os.listdir(self.image_dir) if name.endswith(".png"))
        limit = int(os.environ.get("OASIS_MAX_SAMPLES", "0"))
        if limit > 0:
            self.names = self.names[:limit]
        if not self.names:
            raise FileNotFoundError(self.image_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, name)).convert("L")
        image, label = self.transforms(image, label)
        return {"image": image, "label": label, "name": name}

    def transforms(self, image, label):
        width = self.opt.load_size
        image = TR.functional.resize(image, (width, width), Image.BICUBIC)
        label = TR.functional.resize(label, (width, width), Image.NEAREST)
        crop_x = random.randint(0, max(0, width - self.opt.crop_size))
        crop_y = random.randint(0, max(0, width - self.opt.crop_size))
        box = (crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size)
        image, label = image.crop(box), label.crop(box)
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics) and random.random() < 0.5:
            image, label = TR.functional.hflip(image), TR.functional.hflip(label)
        image = TR.functional.normalize(TR.functional.to_tensor(image), (0.5,) * 3, (0.5,) * 3)
        label = torch.from_numpy(np.asarray(label, dtype=np.int64)).unsqueeze(0).float() / 255.0
        return image, label
