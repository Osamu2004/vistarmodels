"""One isolated setup/inference/scoring lifecycle, with a fresh launch gate."""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from campus_prepare import TASK, prepare, sha

CODE = Path(__file__).resolve().parent
cache = Path(os.environ['TMPDIR']).parent
epoch_file=TASK/'EXECUTION_EPOCH.json'
if not epoch_file.exists():
    epoch_file.write_text(json.dumps(dict(started_epoch=time.time()))+'\n')
started=json.loads(epoch_file.read_text())['started_epoch']


def write(name, value):
    (TASK / name).write_text(json.dumps(value, indent=2) + '\n')


def execute(args, env, label):
    with (TASK / (label + '.log')).open('w') as f:
        p = subprocess.Popen(list(map(str, args)), env=env, stdout=f, stderr=subprocess.STDOUT)
        write('active_process.json', dict(pid=p.pid, phase=label, started_epoch=time.time(),
                                         gpu_uuid=env.get('CUDA_VISIBLE_DEVICES'), argv=list(map(str,args))))
        print('START', label, p.pid, flush=True)
        code = p.wait()
    write(label + '.terminal.json', dict(pid=p.pid, exit_code=code, ended_epoch=time.time()))
    if code:
        raise RuntimeError(f'{label} exit {code}')


try:
    if '--runtime-ready' in sys.argv:
        ready = json.loads((TASK / 'READY.json').read_text())
    else:
        ready = prepare(cache)
        os.execve(ready['python'], [ready['python'], '-u', str(Path(__file__).resolve()),
                                  '--runtime-ready'], dict(os.environ, **ready['env']))
    cpu_env = dict(os.environ, **ready['env'])
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    execute([ready['python'], '-m', 'pip', 'install', '--no-compile',
             '--index-url', 'https://pypi.tuna.tsinghua.edu.cn/simple',
             'fvcore==0.1.5.post20221221', 'iopath', 'termcolor'],
            cpu_env, 'cpu_model_dependencies')
    execute([ready['python'], '-c',
             'import torch,selective_scan,selective_scan_cuda_core,fvcore,easydict;print(torch.__version__,selective_scan_cuda_core.__file__)'],
            cpu_env, 'cpu_extension_imports')
    env = dict(os.environ, **ready['env'])
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(CODE), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.update(TRITON_CACHE_DIR=str(cache/'scratch/triton'),
               CUDA_CACHE_PATH=str(cache/'scratch/cuda_cache'),
               TORCH_EXTENSIONS_DIR=str(cache/'scratch/torch_extensions'),
               TORCH_HOME=str(TASK/'cache/torch'), XDG_CACHE_HOME=str(TASK/'cache/xdg'),
               MPLCONFIGDIR=str(TASK/'cache/matplotlib'))
    py = ready['python']
    results = []
    # Frozen before seeing test metrics. The primary follows official resize512;
    # the secondary repeats the previous tile512 spatial protocol in FP32.
    for label, tile, maximum in [('smoke2', 1024, 2), ('whole_resize512', 1024, 0), ('native_tile512', 512, 0)]:
        gate = TASK / ('LAUNCH_' + label + '.json')
        waiting_since=time.time()
        write('AWAITING_GPU.json',dict(phase=label,requested_epoch=waiting_since,
                                        gate=str(gate),cuda_started=False))
        while not gate.exists():
            if time.time()-waiting_since>7200:
                raise TimeoutError('No coordinated launch authorization for '+label)
            time.sleep(2)
        auth=json.loads(gate.read_text())
        assert 0 <= time.time()-auth['created_epoch'] < 45
        assert auth['phase']==label and auth['reservation_id'] and auth['gpu_uuid'].startswith('GPU-')
        env['CUDA_VISIBLE_DEVICES']=auth['gpu_uuid']
        out = cache / 'scratch/results' / label
        args = [py, '-u', CODE/'campus_infer.py', '--data_root', ready['data_root'],
                '--split', 'test', '--output_dir', out, '--rcdnet_root', ready['vendor'],
                '--checkpoint', ready['checkpoint'], '--clip_model', ready['clip'],
                '--prompt', 'building', '--threshold', '0.5', '--tile_size', str(tile),
                '--model_input_size', '512', '--amp', 'fp32', '--save_images']
        if maximum:
            args += ['--max_samples', str(maximum)]
        execute(args, env, label)
        if maximum:
            assert json.loads((out/'metrics.json').read_text())['num_evaluated_pairs'] == 2
            continue
        execute([py, '-u', CODE/'tools/score_rcdnet_boundary.py', '--eval_dir', out],
                dict(env, CUDA_VISIBLE_DEVICES=''), label + '_boundary')
        summary = json.loads((out/'boundary_wfm_3px/paper_summary.json').read_text())
        assert summary['complete'] and summary['processed_samples'] == 128
        dest = TASK / 'evaluations' / label
        dest.mkdir(parents=True, exist_ok=False)
        for name in ['metrics.json', 'per_image_metrics.jsonl', 'boundary_wfm_3px/paper_summary.json', 'boundary_wfm_3px/per_image_boundary.json']:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((out/name).read_bytes())
        for name in ['runtime_diagnostics.json','model_grid512_metrics.json']:
            if (out/name).exists():
                (dest/name).write_bytes((out/name).read_bytes())
        archive = dest / 'predictions_and_gt.tar'
        folders=['pred_mask','gt_mask']
        if (out/'pred_grid512').is_dir():
            folders+=['pred_grid512','gt_grid512']
        with tarfile.open(archive, 'w') as tf:
            for folder in folders:
                files = sorted((out/folder).glob('*.png'))
                assert len(files) == 128
                for path in files:
                    tf.add(path, arcname=f'{folder}/{path.name}', recursive=False)
        with tarfile.open(archive) as tf:
            assert len(tf.getmembers()) == 128*len(folders)
            from PIL import Image
            import numpy as np
            for m in tf:
                with Image.open(io.BytesIO(tf.extractfile(m).read())) as image:
                    a = np.asarray(image)
                    size=512 if '_grid512/' in m.name else 1024
                    assert a.shape == (size,size) and set(np.unique(a)) <= {0,255}
        grid = json.loads((out/'model_grid512_metrics.json').read_text()) if (out/'model_grid512_metrics.json').exists() else None
        results.append(dict(protocol=label, samples=128, num_failed=0,
                            input_grid512_iou_percent=100*grid['iou_change'] if grid else None,
                            iou_change_percent=100*summary['iou_verification']['iou_change'],
                            wfm_3px_percent=summary['scores']['wfm_3px_percent'],
                            artifact_sha256=sha(archive), artifact_bytes=archive.stat().st_size))
        write('RESULTS_PARTIAL.json', results)
    assert sha(ready['checkpoint']) == ready['checkpoint_sha256']
    write('RESULT.json', dict(complete=True, results=results, primary_protocol='whole_resize512',
                             checkpoint_sha256=ready['checkpoint_sha256'],
                             official_revision=ready['official_revision'],
                             elapsed_seconds=time.time()-started))
    print('COMPLETE', json.dumps(results), flush=True)
except BaseException as e:
    write('FAILURE.json', dict(error=repr(e), elapsed_seconds=time.time()-started))
    raise
