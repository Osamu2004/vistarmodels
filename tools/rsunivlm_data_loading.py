"""Ordered CPU RGB prefetch for the unchanged RSUniVLM evaluation loops.

The Dataset delegates decoding to the original protocol. Labels, saved answers,
mask validation, cropping, query grouping and GPU calls stay in the evaluator.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LoaderOptions:
    num_workers: int = 2
    prefetch_factor: int = 2
    timeout: float = 120.0

    def __post_init__(self):
        if self.num_workers < 0 or self.prefetch_factor < 1 or not np.isfinite(self.timeout) or self.timeout < 0:
            raise ValueError('DataLoader requires workers>=0, prefetch>=1 and finite timeout>=0')

    @classmethod
    def from_env(cls):
        return cls(num_workers=int(os.environ.get('RSUNIVLM_NUM_WORKERS', '2')),
                   prefetch_factor=int(os.environ.get('RSUNIVLM_PREFETCH_FACTOR', '2')),
                   timeout=float(os.environ.get('RSUNIVLM_LOADER_TIMEOUT', '120')))


def identity_collate(item):
    # Automatic batching is disabled: each item is one full image or image pair.
    return item


def cpu_worker_init(_worker_id):
    # Called only inside spawned loader workers, never in the GPU model process.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass


class RGBSourceDataset(Dataset):
    def __init__(self, dataset, rows):
        self.dataset = dataset
        # Only manifest records cross spawn boundaries; never a model or reader closure.
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from tools import rsunivlm_paper_protocol as protocol
        row = self.rows[index]
        sample = protocol.Sample.restore(row['sample'])
        started = time.monotonic()
        if protocol.metadata(sample) != row['files']:
            raise ValueError(f'Inputs changed before RGB prefetch: {sample.name}')
        images = protocol.load_images(sample, self.dataset)
        try:
            arrays = []
            for image in images:
                array = np.asarray(image)
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError(f'Original decoder did not return RGB uint8: {sample.name}')
                # CPU uint8 tensors use DataLoader's shared-memory transport.
                arrays.append(torch.from_numpy(np.array(array, copy=True, order='C')))
            return dict(index=index, name=sample.name, rgb=arrays,
                        decode_seconds=time.monotonic()-started)
        finally:
            for image in images:
                image.close()


def make_loader(dataset, rows, options):
    # A dedicated CPU generator prevents DataLoader's base seed from consuming
    # the model process's default torch RNG state during construction/iteration.
    generator = torch.Generator(device='cpu').manual_seed(0)
    kwargs = dict(dataset=RGBSourceDataset(dataset, rows), batch_size=None,
                  shuffle=False, drop_last=False, num_workers=options.num_workers,
                  collate_fn=identity_collate, pin_memory=False, generator=generator,
                  persistent_workers=False,
                  timeout=options.timeout if options.num_workers else 0)
    if options.num_workers:
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=options.prefetch_factor,
                      worker_init_fn=cpu_worker_init)
        if 'in_order' in inspect.signature(DataLoader).parameters:
            kwargs['in_order'] = True
    return DataLoader(**kwargs)


class OrderedRGBReader:
    def __init__(self, entry, plan, rank, world_size, options):
        self.dataset = entry['dataset']
        selected = entry['samples'][:plan['max_samples']] if plan['max_samples'] else entry['samples']
        self.rows = selected[rank::world_size]
        self.options = options
        self.loader = make_loader(self.dataset, self.rows, options)
        self.iterator = None
        self.position = 0
        self.wait_seconds = self.decode_seconds = 0.0
        self.cleanup_error = None

    def read(self, sample, dataset):
        from tools import rsunivlm_paper_protocol as protocol
        if dataset != self.dataset or self.position >= len(self.rows):
            raise ValueError('RGB DataLoader was requested outside its assigned dataset/shard')
        row = self.rows[self.position]
        if sample != protocol.Sample.restore(row['sample']):
            raise ValueError(f'RGB request order differs from the original evaluator: {sample.name}')
        if protocol.metadata(sample) != row['files']:
            raise ValueError(f'Inputs changed after RGB prefetch: {sample.name}')
        started = time.monotonic()
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            item = next(self.iterator)
        except StopIteration as error:
            raise ValueError('RGB DataLoader ended before its assigned source population') from error
        self.wait_seconds += time.monotonic()-started
        if item['index'] != self.position or item['name'] != sample.name:
            raise ValueError('DataLoader changed the source order; refusing to change query batches')
        images = []
        try:
            for value in item['rgb']:
                if value.device.type != 'cpu' or value.dtype != torch.uint8 or value.ndim != 3 or value.shape[2] != 3:
                    raise ValueError('Expected a CPU uint8 RGB tensor from the DataLoader')
                images.append(Image.fromarray(value.numpy()))
        except BaseException:
            for image in images:
                image.close()
            raise
        self.decode_seconds += item['decode_seconds']
        self.position += 1
        return images

    def close(self):
        iterator, self.iterator = self.iterator, None
        self.loader = None
        # DataLoader has no public close method. Shut its iterator down explicitly
        # on a partial/failed pass; compatibility is guarded for num_workers=0.
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception as error:
                self.cleanup_error = f'{type(error).__name__}: {error}'

    def statistics(self):
        return dict(dataset=self.dataset, assigned_sources=len(self.rows), consumed_sources=self.position,
                    rgb_wait_seconds=self.wait_seconds, consumed_worker_decode_seconds=self.decode_seconds,
                    options=asdict(self.options), prefetch_sources_per_gpu=self.options.num_workers*self.options.prefetch_factor,
                    multiprocessing_context='spawn' if self.options.num_workers else 'single_process',
                    automatic_source_batching=False, shuffle=False, drop_last=False, pin_memory=False,
                    persistent_workers=False, cleanup_error=self.cleanup_error,
                    note='CPU source images only; original tile/query/GPU batch/cache/GT/scoring logic retained')


class _ProtocolReaderProxy:
    def __init__(self, original, reader):
        self.original, self.reader = original, reader

    def __getattr__(self, name):
        return getattr(self.original, name)

    def load_images(self, sample, dataset):
        return self.reader.read(sample, dataset)


@contextmanager
def bind_ordered_reader(engine, entry, output, options):
    from tools import rsunivlm_paper_worker as worker
    from tools import rsunivlm_native_semantic as native
    from baselines.rsunivlm import evaluate as storage
    reader = OrderedRGBReader(entry, engine.plan, engine.rank, engine.world_size, options)
    worker_protocol, native_protocol = worker.protocol, native.protocol
    # Replace only the evaluator globals. The real protocol module stays intact
    # for spawned DataLoader workers, including the num_workers=0 decode path.
    worker.protocol = _ProtocolReaderProxy(worker_protocol, reader)
    native.protocol = _ProtocolReaderProxy(native_protocol, reader)
    try:
        print(f'[RGB DataLoader] GPU{engine.rank} {entry["dataset"]}: workers/GPU={options.num_workers}, '
              f'prefetch/worker={options.prefetch_factor}, sources={len(reader.rows)}, ordered=True', flush=True)
        yield reader
    finally:
        worker.protocol, native.protocol = worker_protocol, native_protocol
        reader.close()
        report = dict(identity=engine.plan['identity'], rank=engine.rank,
                      loader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **reader.statistics())
        path = Path(output)/entry['dataset']/'shards'/f'rank{engine.rank:03d}'/'data_loading.json'
        try:
            storage.write_json(path, report)
        except OSError as error:
            # Optional I/O timing must not replace an original CUDA exception.
            print(f'[RGB DataLoader] Cannot save timing metadata: {error}', flush=True)
        print(f'[RGB DataLoader] GPU{engine.rank} {entry["dataset"]}: consumed={reader.position}, '
              f'wait={reader.wait_seconds:.2f}s, worker_decode={reader.decode_seconds:.2f}s', flush=True)


def install_data_loading(engine_class, options):
    original_evaluate = engine_class.evaluate

    def evaluate(self, entry, output):
        with bind_ordered_reader(self, entry, output, options):
            return original_evaluate(self, entry, output)

    engine_class.evaluate = evaluate
