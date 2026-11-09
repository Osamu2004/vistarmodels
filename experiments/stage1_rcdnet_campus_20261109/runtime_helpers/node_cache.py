"""Opt-in node-local cache with bounded outbox and verified copy-back.

Standard-library Python 3.9+. Only directories created by this manager are
managed. Producers must use stage_files(), keep the consumer marker, and never
use an outbox file after staging. This is an application budget, not a kernel
quota for arbitrary third-party writes. Existing experiments are never adopted.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from experiment_local_cache import cache_consumers, remove_owned_tree

GIB = 1024**3
TASK_CACHE_MAX = 8 * GIB
POLICY_REVISION = 'task-cache-8gib-direct-v1'
SCHEMA = 'campus_node_cache_v1'
ROOT = Path('/tmp') / ('campus-gpu-cache-' + str(os.getuid()))
SAFE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
META_ALLOWANCE = 64 * 1024


class Backpressure(RuntimeError):
    pass


class BoundedOutput:
    """File-like writer for np.save and streaming output; no raw fileno escape."""
    def __init__(self,path,size):
        self._file=regular_path(path).open('xb');self._size=size;self._written=0
    def write(self,data):
        if len(data)>self._size-self._written:raise Backpressure('Output would exceed its reserved length')
        count=self._file.write(data);self._written+=count;return count
    def tell(self):return self._written
    def flush(self):return self._file.flush()
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):
        try:
            self._file.flush();os.fsync(self._file.fileno())
            if kind is None and self._written!=self._size:raise ValueError('Output shorter than reserved length')
        finally:self._file.close()


@dataclass(frozen=True)
class Policy:
    node_bytes: int = 64 * GIB
    run_bytes: int = TASK_CACHE_MAX
    free_bytes: int = 50 * GIB
    free_fraction: float = .1
    max_chunks: int = 32


def regular_path(path):
    p = Path(os.path.abspath(path))
    if any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('Symlink path rejected: ' + str(p))
    return p


def read(path):
    p = regular_path(path)
    if not p.is_file() or p.stat().st_size > 8 * 1024**2:
        raise ValueError('Missing or oversized cache metadata')
    return json.loads(p.read_text())


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    path = regular_path(path)
    temporary = regular_path(path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.pending'))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path); sync_dir(path.parent)
    finally:
        if temporary.exists(): temporary.unlink()


def sha(path):
    h = hashlib.sha256()
    with regular_path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not SAFE.fullmatch(value):
        raise ValueError('Invalid cache identifier')
    return value


def label(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 500 or any(ord(c) < 32 for c in value):
        raise ValueError('Invalid project/task/owner label')
    return value


def proc_identity(pid):
    p = Path('/proc') / str(pid)
    if not Path('/proc').is_dir():
        try:
            os.kill(pid, 0)
            return {'pid': pid, 'presence': 'alive', 'uid': os.getuid(), 'start_ticks': None, 'boot_id': None}
        except ProcessLookupError:
            return {'pid': pid, 'presence': 'gone'}
    try:
        s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'presence': 'gone' if s[0] == 'Z' else 'alive',
                'uid': p.stat().st_uid, 'start_ticks': int(s[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return {'pid': pid, 'presence': 'gone'}
    except (OSError, ValueError, IndexError):
        return {'pid': pid, 'presence': 'unknown'}


def alive(identity):
    if not identity:
        return False
    current = proc_identity(identity['pid'])
    if current['presence'] == 'gone':
        return False
    if current['presence'] == 'unknown':
        return True
    return all(current.get(k) == identity.get(k) for k in ('uid', 'start_ticks', 'boot_id'))


def allocated(path):
    total = 0
    for current, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            p = Path(current) / name
            # Account link inodes without following their external targets.
            total += p.lstat().st_blocks * 512
    return total


def directory_identity(path):
    s = path.stat()
    ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf('SC_CLK_TCK')) if hasattr(time, 'CLOCK_BOOTTIME') else None
    boot = Path('/proc/sys/kernel/random/boot_id')
    return {'schema': SCHEMA, 'cache': str(path), 'uid': os.getuid(), 'token': uuid.uuid4().hex,
            'device': s.st_dev, 'inode': s.st_ino, 'created_boot_ticks': ticks,
            'boot_id': boot.read_text().strip() if boot.exists() else None}


class Store:
    def __init__(self, root=ROOT, policy=None):
        self.root = regular_path(root)
        self.policy = policy or Policy()

    @contextmanager
    def ledger(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        s = self.root.stat()
        if s.st_uid != os.getuid() or s.st_mode & 0o077:
            raise ValueError('Manager directory must be private and owned')
        fd = os.open(self.root / 'LOCK', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            p = self.root / 'registry.json'
            data = read(p) if p.exists() else {'schema': SCHEMA, 'runs': {}}
            if data.get('schema') != SCHEMA:
                raise ValueError('Unsupported cache ledger')
            before=json.dumps(data,sort_keys=True)
            yield data
            ended = sorted((r for r in data['runs'].values() if r['state'] == 'reclaimed'), key=lambda r: r['updated'])
            for r in ended[:-100]:
                del data['runs'][r['id']]
            if not p.exists() or json.dumps(data,sort_keys=True)!=before:atomic(p, data)
        finally:
            os.close(fd)

    def checked(self, data, run_id, token):
        r = data['runs'][identifier(run_id)]
        if r['token'] != token:
            raise ValueError('Cache ownership token differs')
        if r['state'] != 'reclaimed' and not (r['state']=='reclaiming' and not Path(r['path']).exists()):
            path = regular_path(r['path']); s = path.stat(); i = r['identity']
            if (s.st_uid, s.st_dev, s.st_ino) != (i['uid'], i['device'], i['inode']) or read(path / 'OWNER.json') != i:
                raise ValueError('Managed cache identity changed')
        return r

    def get(self, run_id, token):
        with self.ledger() as d:
            return json.loads(json.dumps(self.checked(d, run_id, token)))

    def floor(self):
        v = os.statvfs(self.root)
        return max(self.policy.free_bytes, int(v.f_blocks * v.f_frsize * self.policy.free_fraction))

    def capacity(self, extra=0):
        return shutil.disk_usage(self.root).free - extra >= self.floor()

    def create(self, owner, project, task, request_id, destination, budget=8*GIB, logical_task_id=None):
        for value in (owner, project, task): label(value)
        logical_task_id = label(logical_task_id if logical_task_id is not None else task)
        identifier(request_id)
        limit = min(TASK_CACHE_MAX, self.policy.run_bytes)
        if type(budget) is not int or not META_ALLOWANCE < budget <= limit:
            raise ValueError('Run cache budget outside policy')
        destination = regular_path(destination)
        if not destination.is_absolute() or destination == self.root or destination.is_relative_to(self.root) or self.root.is_relative_to(destination):
            raise ValueError('Durable results must be separate from cache manager')
        with self.ledger() as d:
            for old in d['runs'].values():
                if (old['owner'], old['request_id']) == (owner, request_id):
                    if (old['budget'], old['destination_parent'], old['project'], old['task'], old.get('logical_task_id', old['task'])) != (budget, str(destination), project, task, logical_task_id):
                        raise ValueError('Request ID was used with different parameters')
                    return old
            task_reserved = sum(r['budget'] for r in d['runs'].values()
                                if r['state'] != 'reclaimed' and (r['owner'], r['project'], r.get('logical_task_id', r['task'])) == (owner, project, logical_task_id))
            if task_reserved + budget > limit:
                raise Backpressure('Logical task cache budget is already allocated; use durable paths')
            reserved = sum(r['budget'] for r in d['runs'].values() if r['state'] != 'reclaimed')
            if sum(r['state']!='reclaimed' for r in d['runs'].values())>=64:
                raise Backpressure('Maximum active/retained cache registrations reached')
            if reserved + budget > self.policy.node_bytes:
                raise Backpressure('Node cache budget is already allocated (retained runs included)')
            # Reserve outstanding budgets as well as current physical usage.
            outstanding = sum(max(0, r['budget'] - allocated(r['path'])) for r in d['runs'].values() if r['state'] != 'reclaimed')
            if not self.capacity(outstanding + budget):
                raise Backpressure('Local disk reserve floor would be crossed')
            run_id = uuid.uuid4().hex
            path = self.root / run_id; path.mkdir(mode=0o700)
            identity = directory_identity(path); atomic(path / 'OWNER.json', identity)
            for folder in ('payload', 'scratch', 'outbox'):
                (path / folder).mkdir(mode=0o700)
            r = {'id': run_id, 'owner': owner, 'project': project, 'task': task, 'request_id': request_id,
                 'token': identity['token'], 'identity': identity, 'path': str(path), 'budget': budget, 'logical_task_id': logical_task_id, 'agent_policy_revision':POLICY_REVISION,
                 'destination_parent': str(destination), 'destination': str(destination / run_id),
                 'state': 'prepared', 'chunks': {}, 'commits': {}, 'committed_bytes': 0,
                 'worker': None, 'exit': None, 'updated': time.time(), 'reason': None}
            d['runs'][run_id] = r
        # Never hold the node allocation lock over NFS I/O.
        try:
            out = regular_path(r['destination']); out.mkdir(mode=0o700, parents=True, exist_ok=False)
            atomic(out / 'RUN.json', {k: r[k] for k in ('id','owner','project','task','request_id','budget')})
        except BaseException:
            with self.ledger() as d:
                d['runs'][run_id].update(state='retained', reason='durable_initialization_failed')
            raise
        return self.get(run_id, r['token'])

    def pressure(self, r, charge=0):
        path = Path(r['path'])
        used = allocated(path)
        reserved = sum(c['charge'] for c in r['chunks'].values())
        actual_outbox = allocated(path / 'outbox')
        charged = max(used, used - actual_outbox + reserved)
        if charged + charge > min(r['budget'], TASK_CACHE_MAX):
            raise Backpressure('Per-run cache budget reached; use durable paths for new output')
        if not self.capacity(charge):
            raise Backpressure('Local disk reserve floor reached')

    def begin(self, run_id, token, key, sizes):
        identifier(key)
        if not isinstance(sizes, dict) or not sizes or len(sizes) > 64:
            raise ValueError('A chunk requires 1..64 named files')
        for name, size in sizes.items():
            identifier(name)
            if name in {'OWNER.json','READY.json','COMMITTED.json','DIRECT.json'} or type(size) is not int or size < 0:
                raise ValueError('Invalid bounded output file')
        charge = sum(((n+4095)//4096)*4096 for n in sizes.values()) + META_ALLOWANCE
        with self.ledger() as d:
            r = self.checked(d, run_id, token)
            if r['exit'] is not None or r['state'] == 'reclaimed': raise ValueError('Producer is already closed')
            if key in r['chunks'] or key in r['commits'] or key in r.get('direct', {}): raise ValueError('Chunk ID already used; never repeat producer')
            if len(r['chunks']) >= self.policy.max_chunks: raise Backpressure('Pending chunk count reached')
            self.pressure(r, charge)
            p = Path(r['path']) / 'outbox' / key; p.mkdir(mode=0o700)
            ident = directory_identity(p); atomic(p / 'OWNER.json', ident)
            r['chunks'][key] = {'key': key, 'state': 'building', 'sizes': sizes, 'charge': charge,
                                'identity': ident, 'created': time.time()}
            r.update(state='running', updated=time.time(), reason=None)
            return str(p)

    def ready(self, run_id, token, key):
        r = self.get(run_id, token)
        if key in r.get('direct', {}):
            return self.direct_ready(run_id, token, key)
        c = r['chunks'][key]
        p = regular_path(Path(r['path']) / 'outbox' / key)
        if c['state'] != 'building': raise ValueError('Only a building chunk can become ready')
        if set(x.name for x in p.iterdir()) != set(c['sizes']) | {'OWNER.json'}:
            raise ValueError('Unregistered files in output chunk')
        files = {}
        for name, size in c['sizes'].items():
            f = regular_path(p / name)
            if not f.is_file() or f.stat().st_size != size: raise ValueError('Output length differs from reservation')
            with f.open('rb') as stream: os.fsync(stream.fileno())
            files[name] = {'bytes': size, 'sha256': sha(f)}
        manifest = {'schema': SCHEMA, 'run_id': run_id, 'key': key, 'files': files}
        atomic(p / 'READY.json', manifest)
        with self.ledger() as d:
            r = self.checked(d, run_id, token); self.pressure(r)
            r['chunks'][key].update(state='ready', manifest=manifest)
            r['updated'] = time.time()
        return manifest

    def output(self,run_id,token,key,name):
        r=self.get(run_id,token);identifier(key)
        direct=key in r.get('direct',{})
        c=r['direct'][key] if direct else r['chunks'][key]
        if c['state']!='building' or name not in c['sizes']:raise ValueError('Output must have a prior building reservation')
        folder=Path(r['destination'])/key if direct else Path(r['path'])/'outbox'/key
        if direct and read(folder/'DIRECT.json') != c['marker']:
            raise ValueError('Direct output identity changed')
        return BoundedOutput(folder/name,c['sizes'][name])

    def route(self, run_id, token, key, sizes):
        """Choose the whole output unit before forward; a full cache never waits."""
        try:
            path = self.begin(run_id, token, key, sizes)
            return {'mode':'cache', 'path':path, 'reason':None}
        except Backpressure as error:
            reason = str(error)
        # begin validates identifiers/sizes before it can raise Backpressure.
        # Record a unique direct slot before touching NFS or executing a forward.
        with self.ledger() as d:
            r = self.checked(d, run_id, token)
            if r['exit'] is not None or r['state'] == 'reclaimed':
                raise ValueError('Producer is already closed')
            if key in r['chunks'] or key in r['commits'] or key in r.get('direct', {}):
                raise ValueError('Chunk ID already used; never repeat producer')
            r.setdefault('direct', {})[key] = {'state':'preparing', 'sizes':dict(sizes), 'reason':reason}
            r['fallback_count'] = r.get('fallback_count', 0) + 1
            r['updated'] = time.time()
            destination = Path(r['destination']) / key
        marker = {'schema':SCHEMA, 'run_id':run_id, 'key':key, 'sizes':sizes, 'reason':reason, 'mode':'direct_data1'}
        try:
            if read(Path(r['destination'])/'RUN.json').get('id') != run_id:
                raise ValueError('Durable run identity changed')
            destination = regular_path(destination)
            destination.mkdir(mode=0o700, exist_ok=False)
            atomic(destination/'DIRECT.json', marker)
            sync_dir(destination.parent)
            with self.ledger() as d:
                current = self.checked(d, run_id, token)
                current['direct'][key].update(state='building', marker=marker)
                current['updated'] = time.time()
        except BaseException as error:
            with self.ledger() as d:
                self.checked(d, run_id, token).update(state='retained', reason='direct_initialization_failed: '+str(error)[:200], updated=time.time())
            raise
        return {'mode':'direct_data1', 'path':str(destination), 'reason':reason}

    def direct_ready(self, run_id, token, key):
        """Verify direct NFS bytes and commit without any local payload copy."""
        r = self.get(run_id, token); c = r['direct'][key]
        if c['state'] != 'building':
            raise ValueError('Direct output is not initialized')
        p = regular_path(Path(r['destination'])/key)
        expected = set(c['sizes']) | {'DIRECT.json'}
        if (set(x.name for x in p.iterdir()) - {'COMMITTED.json'} != expected
                or read(p/'DIRECT.json') != c['marker']
                or read(Path(r['destination'])/'RUN.json').get('id') != run_id):
            raise ValueError('Direct output identity or files changed')
        files = {}
        for name, size in c['sizes'].items():
            f = regular_path(p/name)
            if not f.is_file() or f.stat().st_size != size:
                raise ValueError('Direct output length differs from declaration')
            with f.open('rb') as stream: os.fsync(stream.fileno())
            files[name] = {'bytes':size, 'sha256':sha(f)}
        manifest = {'schema':SCHEMA, 'run_id':run_id, 'key':key, 'files':files}
        receipt = p/'COMMITTED.json'
        if receipt.exists():
            if read(receipt) != manifest: raise ValueError('Conflicting direct receipt; preserve all bytes')
        else: atomic(receipt, manifest)
        sync_dir(p); sync_dir(p.parent)
        receipt_sha = sha(receipt)
        with self.ledger() as d:
            current = self.checked(d, run_id, token)
            if key not in current['direct'] or key in current['commits']:
                raise ValueError('Direct output was already committed')
            count = sum(v['bytes'] for v in files.values())
            current['commits'][key] = {'receipt_sha256':receipt_sha, 'bytes':count, 'mode':'direct_data1'}
            current['committed_bytes'] += count
            current['direct_completed_chunks'] = current.get('direct_completed_chunks', 0) + 1
            current['direct_bytes'] = current.get('direct_bytes', 0) + count
            del current['direct'][key]
            current['updated'] = time.time()
        return manifest

    def wait_begin(self, run_id, token, key, sizes, timeout=300):
        """Compatible path-returning entry; full units now go direct, never wait."""
        return self.route(run_id, token, key, sizes)['path']

    def stage_files(self, run_id, token, key, sources):
        sources = {identifier(k): regular_path(v) for k, v in sources.items()}
        if any(not v.is_file() for v in sources.values()): raise ValueError('Regular source files required')
        sizes = {k: v.stat().st_size for k, v in sources.items()}
        folder = Path(self.begin(run_id, token, key, sizes))
        for name, source in sources.items():
            before = source.stat()
            with source.open('rb') as src, self.output(run_id,token,key,name) as dst:
                left = sizes[name]
                while left:
                    block = src.read(min(left, 1024**2))
                    if not block: raise ValueError('Source shortened during staging')
                    dst.write(block); left -= len(block)
                if src.read(1): raise ValueError('Source grew during staging')
            after = source.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Source changed during staging')
        return self.ready(run_id, token, key)

    def copy_one(self, run_id, token):
        # The per-run flock prevents duplicate writers and is released on crash.
        r = self.get(run_id, token)
        if r['state'] in ('reclaimed','reclaiming'): return False
        lock = os.open(Path(r['path']) / 'COPY.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return False
            with self.ledger() as d:
                r = self.checked(d, run_id, token)
                candidates = [c for c in r['chunks'].values() if c['state'] in ('ready','copying','copy_failed','committed')]
                if not candidates: return False
                c = min(candidates, key=lambda c: c['created']); key = c['key']
                prior = c['state']
                if prior != 'committed': c['state'] = 'copying'
                r.update(updated=time.time(), reason=None)
                c = json.loads(json.dumps(c))
            try:
                source = regular_path(Path(r['path']) / 'outbox' / key)
                manifest = c['manifest'] if prior=='committed' else read(source / 'READY.json')
                if manifest != c['manifest']: raise ValueError('Ready output manifest changed')
                if read(Path(r['destination'])/'RUN.json').get('id') != run_id: raise ValueError('Durable run identity changed')
                destination = regular_path(Path(r['destination']) / key)
                destination.mkdir(mode=0o700, exist_ok=True)
                # Result directory is unique to this run. Existing bytes must match;
                # never truncate or replace partial/foreign results.
                for name, info in manifest['files'].items():
                    src = regular_path(source / name); dst = regular_path(destination / name)
                    if prior=='committed':
                        if sha(destination/'COMMITTED.json')!=r['commits'][key]['receipt_sha256']:
                            raise ValueError('Committed receipt changed during recovery')
                        continue
                    if sha(src) != info['sha256'] or src.stat().st_size != info['bytes']:
                        raise ValueError('Local output changed after ready')
                    if not dst.exists():
                        with src.open('rb') as a, dst.open('xb') as b:
                            shutil.copyfileobj(a, b, 1024**2); b.flush(); os.fsync(b.fileno())
                    elif dst.is_file() and dst.stat().st_size < info['bytes']:
                        # Resume only an exact prefix inside this run's private output.
                        with src.open('rb') as a, dst.open('r+b') as b:
                            left=dst.stat().st_size
                            while left:
                                block=b.read(min(left,1024**2))
                                if not block or a.read(len(block))!=block:raise ValueError('Partial durable prefix differs')
                                left-=len(block)
                            shutil.copyfileobj(a,b,1024**2);b.flush();os.fsync(b.fileno())
                    if not dst.is_file() or dst.stat().st_size != info['bytes'] or sha(dst) != info['sha256']:
                        raise ValueError('Conflicting or incomplete durable output; preserved for recovery')
                receipt = destination / 'COMMITTED.json'
                if receipt.exists():
                    if read(receipt) != manifest: raise ValueError('Conflicting durable receipt')
                else: atomic(receipt, manifest)
                sync_dir(destination); sync_dir(destination.parent)
                digest = sha(receipt)
                with self.ledger() as d:
                    current = self.checked(d, run_id, token)
                    if key not in current['commits']:
                        current['commits'][key] = {'receipt_sha256': digest, 'bytes': sum(x['bytes'] for x in manifest['files'].values())}
                        current['committed_bytes'] += current['commits'][key]['bytes']
                    current['chunks'][key]['state']='committed'
                # Commit is recorded BEFORE deletion, making either crash window recoverable.
                if source.exists():remove_owned_tree(source,c['identity'])
                with self.ledger() as d:
                    current=self.checked(d,run_id,token)
                    del current['chunks'][key]
                    current.update(updated=time.time(), state='draining' if current['exit'] is not None else 'running', reason=None)
                return True
            except BaseException as error:
                with self.ledger() as d:
                    current = self.checked(d, run_id, token)
                    if key in current['chunks'] and current['chunks'][key]['state']!='committed': current['chunks'][key]['state'] = 'copy_failed'
                    current.update(state='retained', reason=type(error).__name__ + ': ' + str(error)[:240], updated=time.time())
                raise
        finally:
            os.close(lock)

    def bind(self, run_id, token, pid):
        identity = proc_identity(pid)
        if identity['presence'] == 'unknown' or identity.get('uid', os.getuid()) != os.getuid():
            raise ValueError('Worker identity cannot be verified')
        with self.ledger() as d:
            r = self.checked(d, run_id, token)
            if r['worker'] is not None: raise ValueError('Cannot replace a cache worker')
            r.update(worker=identity, state='running', updated=time.time())

    def finish(self, run_id, token, exit_code):
        if exit_code is not None and type(exit_code) is not int: raise ValueError('Real exit code required')
        with self.ledger() as d:
            r = self.checked(d, run_id, token)
            if r['state'] == 'reclaimed': return r
            if alive(r['worker']): raise ValueError('Worker is still alive')
            old = r['exit']
            if old is not None and old['code'] != exit_code: raise ValueError('Cannot rewrite the recorded exit')
            r['exit'] = old or {'code': exit_code, 'known': exit_code is not None, 'recorded': time.time()}
            r.update(state='draining', updated=time.time())
        self.persist_terminal(r)
        while self.copy_one(run_id, token): pass
        return self.retire(run_id, token)

    def persist_terminal(self, r):
        """A retained cache must not hide or lose its producer's real exit."""
        value = {'schema':SCHEMA, 'run_id':r['id'], 'exit':r['exit']}
        path = regular_path(Path(r['destination'])/'TERMINAL.json')
        if path.exists():
            if read(path) != value: raise ValueError('Conflicting durable terminal record')
        else: atomic(path, value)

    def retire(self, run_id, token):
        r = self.get(run_id, token)
        if r['state'] == 'reclaimed': return r
        if r['exit'] is None: raise ValueError('No producer terminal record')
        self.persist_terminal(r)
        consumers = [p for p in cache_consumers(Path(r['path']), r['identity']) if p['pid'] != os.getpid()] if Path(r['path']).exists() else []
        reason = 'worker_or_consumer_alive' if alive(r['worker']) or consumers else 'unfinished_chunks' if r['chunks'] else 'unfinished_direct_output' if r.get('direct') else None
        if reason:
            with self.ledger() as d:
                d['runs'][run_id].update(state='retained', reason=reason, updated=time.time())
            return self.get(run_id, token)
        out = regular_path(r['destination'])
        for key, info in r['commits'].items():
            if sha(out / key / 'COMMITTED.json') != info['receipt_sha256']:
                raise ValueError('Durable commit receipt missing or changed')
        if r['state']=='reclaiming':
            if sha(out/'RESULT.json')!=r['result_sha256']:raise ValueError('Cleanup recovery result changed')
        else:
            result = {k: r[k] for k in ('id','owner','project','task','exit','commits','committed_bytes')}
            result.update({k:r.get(k,0) for k in ('fallback_count','direct_completed_chunks','direct_bytes')})
            result.update(schema=SCHEMA, experiment_success=r['exit']['known'] and r['exit']['code']==0, finished=time.time())
            atomic(out / 'RESULT.json', result)
            result_sha=sha(out/'RESULT.json')
            with self.ledger() as d:
                d['runs'][run_id].update(state='reclaiming',result_sha256=result_sha)
        # Recheck liveness immediately before anchored deletion.
        if alive(r['worker']) or [p for p in cache_consumers(Path(r['path']),r['identity']) if p['pid']!=os.getpid()]:
            return self.get(run_id, token)
        if Path(r['path']).exists():remove_owned_tree(Path(r['path']), r['identity'])
        with self.ledger() as d:
            item = d['runs'][run_id]
            item.update(state='reclaimed', reason=None, updated=time.time(), completed_chunks=len(r['commits']), commits={}, chunks={})
        atomic(out / 'CLEANUP.json', {'schema':SCHEMA,'run_id':run_id,'deleted':True,'result_sha256':sha(out/'RESULT.json'),'time':time.time()})
        return self.get(run_id, token)

    def watch(self, run_id, token, once=False):
        delay = 1
        while True:
            r = self.get(run_id, token)
            if r['state'] == 'reclaimed': return r
            try:
                changed = self.copy_one(run_id, token)
                r = self.get(run_id, token)
                if r['exit'] is not None: r = self.retire(run_id, token)
                elif r['worker'] and not alive(r['worker']) and not alive(r.get('controller')):
                    # Never manufacture an exit0 after supervisor/worker crash.
                    r = self.finish(run_id, token, None)
                delay = min(60,delay*2) if not changed and r['state']=='retained' else 1
            except (OSError, ValueError) as error:
                with self.ledger() as d:
                    item=d['runs'][run_id]
                    item.update(state=item['state'] if item['state']=='reclaiming' else 'retained',reason=str(error)[:240],updated=time.time())
                delay = min(60, delay * 2)
            if once or r['state'] == 'reclaimed': return r
            time.sleep(delay)

    def snapshot(self):
        if not self.root.exists():
            return {'schema':SCHEMA,'runs':[],'policy':self.policy.__dict__,'policy_revision':POLICY_REVISION,'managed':False}
        with self.ledger() as d:
            result = []
            for r in d['runs'].values():
                row = {k:r[k] for k in ('id','owner','project','task','state','budget','committed_bytes','updated','reason','destination')}
                row.update(used_bytes=allocated(r['path']) if r['state']!='reclaimed' else 0,
                           pending_chunks=len(r['chunks']),pending_bytes=sum(c['charge'] for c in r['chunks'].values()),
                           completed_chunks=r.get('completed_chunks',len(r['commits'])))
                row.update({k:r.get(k,0) for k in ('fallback_count','direct_completed_chunks','direct_bytes')})
                result.append(row)
            disk=shutil.disk_usage(self.root)
            return {'schema':SCHEMA,'managed':True,'sampled_epoch':time.time(),'policy':self.policy.__dict__,'policy_revision':POLICY_REVISION,
                    'disk_total_bytes':disk.total,'disk_available_bytes':disk.free,'free_floor_bytes':self.floor(),
                    'reserved_bytes':sum(r['budget'] for r in d['runs'].values() if r['state']!='reclaimed'),'runs':result}


def supervise(store, run_id, token, command):
    r=store.get(run_id,token)
    if r['worker'] or r['exit'] is not None: raise ValueError('Existing worker/run cannot be restarted')
    env=dict(os.environ,CAMPUS_CACHE_RUN_ID=run_id,CAMPUS_CACHE_TOKEN=token,CAMPUS_CACHE_ROOT=str(store.root),
             DIRECT3D_EXPERIMENT_CACHE=str(Path(r['path'])/'payload'),TMPDIR=str(Path(r['path'])/'scratch'))
    out=Path(r['destination'])
    with store.ledger() as d:d['runs'][run_id]['controller']=proc_identity(os.getpid())
    with (out/'worker.log').open('xb') as log:
        worker=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        store.bind(run_id,token,worker.pid)
        writer=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'watch',run_id,token],env=env,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        code=worker.wait();log.flush();os.fsync(log.fileno())
    # Record the actual wait() result before waiting for the independent writer.
    with store.ledger() as d:
        d['runs'][run_id].update(exit={'code':code,'known':True,'recorded':time.time()},state='draining',updated=time.time())
    try: writer.wait(timeout=30)
    except subprocess.TimeoutExpired:
        return {'state':'draining','run_id':run_id,'worker_exit':code,'writer_pid':writer.pid,'cache_retained':True}
    return store.retire(run_id,token)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    create=sub.add_parser('create')
    for name in ('owner','project','task','request-id','destination'):create.add_argument('--'+name,required=True)
    create.add_argument('--budget-mib',type=int,default=8192)
    create.add_argument('--logical-task-id')
    for name in ('watch','recover'):
        q=sub.add_parser(name);q.add_argument('run_id');q.add_argument('token');q.add_argument('--once',action='store_true')
    run=sub.add_parser('run');run.add_argument('run_id');run.add_argument('token');run.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();store=Store(Path(os.environ.get('CAMPUS_CACHE_ROOT',ROOT)))
    if a.action=='status':result=store.snapshot()
    elif a.action=='create':result=store.create(a.owner,a.project,a.task,a.request_id,a.destination,a.budget_mib*1024**2,a.logical_task_id)
    elif a.action in ('watch','recover'):result=store.watch(a.run_id,a.token,a.once)
    else:
        command=a.command[1:] if a.command[:1]==['--'] else a.command
        if not command:raise ValueError('Foreground worker command required')
        result=supervise(store,a.run_id,a.token,command)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,Backpressure) as error:
        print(json.dumps({'error':type(error).__name__,'message':str(error)},ensure_ascii=False));sys.exit(1)
