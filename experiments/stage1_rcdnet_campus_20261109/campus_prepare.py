"""Prepare an isolated RCDNet runtime from verified campus assets; CPU only."""
from pathlib import Path
import hashlib
import gzip
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request

ROOT = Path('/data1/zzy26')
TASK = ROOT / 'outputs/vistar_rcdnet_campus_20261109'
REV = '0966e96ff7075476d77442bbf6623ed5086d52da'
BUDGET = 8 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def run(args, env, log):
    print('RUN', ' '.join(map(str, args)), flush=True)
    with (TASK / log).open('w') as f:
        p = subprocess.run(list(map(str, args)), env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{log}: exit {p.returncode}')


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    tmp = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as f:
        shutil.copyfileobj(response, f, 4 * 1024**2)
    tmp.rename(path)


def prepare(cache):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PIP_CONFIG_FILE='/dev/null', PIP_NO_COMPILE='1',
               PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple')
    old = ROOT / 'outputs/vistar_ovcd_remote_20261108'
    receipt = json.loads((old / 'validated_runtime_export.json').read_text())
    archive = Path(receipt['archive'])
    print('VERIFY portable runtime', archive, flush=True)
    proof_path = TASK / 'runtime_archive_verified.json'
    stat = archive.stat()
    identity = {k:getattr(stat,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    if proof.get('identity') == identity and proof.get('sha256') == receipt['sha256']:
        print('REUSE full SHA verification of the unchanged immutable archive', flush=True)
    else:
        if sha(archive) != receipt['sha256']:
            raise ValueError('Portable runtime SHA mismatch')
        after = archive.stat()
        if identity != {k:getattr(after,k) for k in identity}:
            raise ValueError('Portable archive changed during hashing')
        proof_path.write_text(json.dumps(dict(identity=identity,sha256=receipt['sha256']))+'\n')
    print('EXTRACT selected Python runtime and LEVIR original bytes', flush=True)
    prefixes = ('scratch/ovcd-env/', 'scratch/python_headers/usr/include/python3.9/',
                'payload/evaluation_view/LEVIR-CD/')
    excluded = ('mmcv', 'mmengine', 'mmseg', 'matplotlib', 'sklearn', 'skimage',
                'pytorch_lightning', 'lightning', 'torchmetrics', 'cv2', 'opencv',
                'pandas', 'kornia', 'pytest', 'test', 'qt', 'PyQt')
    total = 0
    with archive.open('rb',buffering=8*1024**2) as raw, gzip.GzipFile(fileobj=raw) as decoded, tarfile.open(fileobj=decoded,mode='r|',bufsize=64*1024) as tf:
        for m in tf:
            if not m.name.startswith(prefixes):
                continue
            rel = Path(m.name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe runtime path')
            if '/site-packages/' in m.name:
                package = m.name.split('/site-packages/', 1)[1].split('/')[0]
                if package.startswith(excluded):
                    continue
            total += m.size
            if total > 6800 * 1024**2:
                raise ValueError('Selected runtime exceeds planned cache allowance')
            if m.issym() and Path(m.linkname).is_absolute():
                target = Path(m.linkname)
                if target.is_relative_to(Path(receipt['source_cache'])):
                    target = cache / target.relative_to(receipt['source_cache'])
                    m.linkname = os.path.relpath(target, (cache / m.name).parent)
                elif m.name.startswith('scratch/ovcd-env/bin/python') and target.resolve() == Path('/usr/bin/python3.9').resolve():
                    pass
                else:
                    raise ValueError(f'Unapproved runtime link: {m.name}')
            if m.issym() and not Path(m.linkname).is_absolute():
                target = Path(os.path.normpath(str((cache / m.name).parent / m.linkname)))
                if not target.is_relative_to(cache):
                    raise ValueError('Runtime link escapes cache')
            if m.islnk() and (Path(m.linkname).is_absolute() or '..' in Path(m.linkname).parts):
                raise ValueError('Unsafe hardlink')
            tf.extract(m, cache)
    py = cache / 'scratch/ovcd-env/bin/python'
    print('runtime extracted bytes', total, flush=True)
    env.update(TMPDIR=str(cache / 'scratch'),
               PIP_CACHE_DIR=str(TASK / 'pip_cache'),
               CUDA_HOME='/usr/local/cuda-12.9',
               PATH='/usr/local/cuda-12.9/bin:' + env['PATH'],
               CC='/data1/zzy26/envs/direct3d-python/bin/x86_64-conda-linux-gnu-gcc', CXX='/data1/zzy26/envs/direct3d-python/bin/x86_64-conda-linux-gnu-g++', MAX_JOBS='2',
               CPATH=str(cache / 'scratch/python_headers/usr/include/python3.9'),
               HF_HOME=str(ROOT / 'cache/vistar_rcdnet_20261109/huggingface'),
               HF_ENDPOINT='https://hf-mirror.com', TOKENIZERS_PARALLELISM='false')
    run([py, '-m', 'pip', 'install', '--no-compile', 'transformers==4.30.2',
         'easydict==1.13', 'gdown==5.2.0', 'ninja', 'wheel', 'setuptools<70'],
        env, 'install.log')
    source_archive = ROOT / f'inputs/vistar_rcdnet_20261109/official_{REV}.tar.gz'
    download(f'https://codeload.github.com/yilmazkorkmaz1/referring_change_detection/tar.gz/{REV}', source_archive)
    vendor = cache / 'scratch/vendor/referring_change_detection'
    with tarfile.open(source_archive) as tf:
        for m in tf:
            parts = Path(m.name).parts
            if len(parts) < 2 or parts[1] != 'RCDNet':
                continue
            rel = Path(*parts[1:])
            if '..' in rel.parts or m.issym() or m.islnk():
                raise ValueError('Unexpected official source entry')
            m.name = str(rel)
            tf.extract(m, vendor)
    checkpoint = ROOT / 'weights/rcdnet/SECOND-model.safetensors'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        run([py, '-m', 'gdown', '--continue',
             'https://drive.google.com/uc?id=1yAs4tB3ScrH5oUcvQLlouBXRjz9CmiQ1',
             '--output', str(checkpoint) + '.part'], env, 'checkpoint_download.log')
        Path(str(checkpoint) + '.part').rename(checkpoint)
    scan = vendor / 'RCDNet/models/encoders/selective_scan'
    run([py, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation',
         '--no-compile', '-e', scan], env, 'selective_scan_build.log')
    clip = ROOT / 'weights/rcdnet/clip-vit-base-patch32'
    deadline = time.time()+1800
    while not (TASK/'HF_CACHE_READY.json').exists():
        if time.time()>deadline:
            raise TimeoutError('Verified official offline CLIP snapshot not ready')
        time.sleep(2)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    code = "from transformers import CLIPTextModel,CLIPTokenizer; import sys; p=sys.argv[1]; kw=dict(revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',local_files_only=True); CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32',**kw).save_pretrained(p); CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32',**kw).save_pretrained(p)"
    run([py, '-u', '-c', code, clip], env, 'clip_download.log')
    with (TASK / 'environment.freeze.txt').open('w') as f:
        subprocess.run([py, '-m', 'pip', 'freeze'], env=env, stdout=f, check=True)
    view = cache / 'payload/evaluation_view/LEVIR-CD'
    from_count = len(list((view / 'test/A').glob('*.png')))
    if from_count != 128:
        raise ValueError(f'Expected 128 original LEVIR pairs, got {from_count}')
    # Cache immutable model inputs only if the whole group fits, with output
    # headroom. Full groups fall back to their persistent paths without waiting.
    inputs=[checkpoint]+[p for p in clip.iterdir() if p.is_file()]
    needed=sum(p.stat().st_size for p in inputs)
    used=sum((Path(parent)/name).lstat().st_blocks*512
             for parent,dirs,files in os.walk(cache,followlinks=False)
             for name in dirs+files)
    cached=False
    if used+needed+128*1024**2 > BUDGET:
        print('MODEL INPUTS direct_data1: cache headroom insufficient',flush=True)
    else:
        model_root=cache/'scratch/model_inputs';model_root.mkdir()
        local_clip=model_root/'clip-vit-base-patch32';local_clip.mkdir()
        for source in inputs:
            target=model_root/source.name if source==checkpoint else local_clip/source.name
            before=source.stat();shutil.copyfile(source,target);after=source.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        checkpoint=model_root/checkpoint.name;clip=local_clip;cached=True
        assert sha(checkpoint)=='f4b772d1a22334a25d9435f4f32a0e0c3db18111e9454bd92586e55535f24a4e'
    print('MODEL INPUT CACHE',cached,'bytes',needed,flush=True)
    result = dict(python=str(py), vendor=str(vendor), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), clip=str(clip), data_root=str(view),
                  original_data_root=str(ROOT / 'data/LEVIR-CD'),
                  official_revision=REV, source_archive_sha256=sha(source_archive),
                  restored_bytes=total, cached_model_inputs=cached,
                  model_input_cache_bytes=needed if cached else 0, created_epoch=time.time(),
                  env={k: env[k] for k in ('TMPDIR','PIP_CACHE_DIR','CUDA_HOME','PATH','CC','CXX',
                       'MAX_JOBS','CPATH','HF_HOME','HF_ENDPOINT','TOKENIZERS_PARALLELISM',
                       'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                       'PYTHONDONTWRITEBYTECODE')})
    (TASK / 'READY.json').write_text(json.dumps(result, indent=2) + '\n')
    print('READY for coordinated CUDA launch', flush=True)
    return result
