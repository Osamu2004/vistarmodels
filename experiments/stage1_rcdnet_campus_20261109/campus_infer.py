"""Use the unmodified RCDNet adapter with compact binary-mask I/O."""
import os
from pathlib import Path
import json
import sys
import time
from PIL import Image
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import baselines.levircd_common as common


def masks(directories, stem, image_a, image_b, target, prediction, *, save_images, **kwargs):
    if not save_images:
        return
    Image.fromarray((target * 255).astype(np.uint8)).save(directories['gt_mask'] / f'{stem}.png')
    Image.fromarray((prediction * 255).astype(np.uint8)).save(directories['pred_mask'] / f'{stem}_pred_mask.png')


common.save_artifacts = masks
from baselines.rcdnet import eval_rcdnet_levircd as adapter
import torch


def option(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


out = Path(option('--output_dir'))
grid_rows = []
if option('--tile_size') == '1024':
    _, samples = common.discover_samples(option('--data_root'), 'test',
                                         int(option('--max_samples', '0')))
    original_build = adapter._build_model

    def capture(model, inputs, logits):
        # A read-only output hook records the SAME forward at its original grid.
        sample = samples[len(grid_rows)]
        pred = (logits.detach().sigmoid()[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        target = common.load_binary(sample.label)
        target = np.asarray(Image.fromarray(target).resize((512,512), Image.Resampling.NEAREST))
        assert pred.shape == target.shape == (512,512)
        counts = common.binary_counts(pred, target)
        grid_rows.append(dict(name=sample.stem, **counts))
        for folder, array, name in [('pred_grid512', pred, sample.stem+'_pred_mask.png'),
                                    ('gt_grid512', target, sample.stem+'.png')]:
            (out/folder).mkdir(parents=True,exist_ok=True)
            Image.fromarray((array*255).astype(np.uint8)).save(out/folder/name)

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        result[0].register_forward_hook(capture)
        return result

    adapter._build_model = build

started = time.monotonic()
adapter.main()
if grid_rows:
    assert len(grid_rows) == len(samples)
    totals = {key:sum(row[key] for row in grid_rows) for key in ('tp','tn','fp','fn')}
    common.write_json(out/'model_grid512_metrics.json',
        dict(scoring_grid=512, num_evaluated_pairs=len(grid_rows),
             protocol='Same whole-resize512 FP32 forward; nearest-resized GT; fixed threshold0.5',
             **totals, **common.binary_metrics(totals), per_image_counts=grid_rows))
common.write_json(out/'runtime_diagnostics.json',
    dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(0), elapsed_seconds=time.monotonic()-started,
         peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
         peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
         matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
         cudnn_tf32=torch.backends.cudnn.allow_tf32))
