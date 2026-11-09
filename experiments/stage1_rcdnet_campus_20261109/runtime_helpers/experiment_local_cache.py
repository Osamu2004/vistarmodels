"""Pack unchanged small-file inputs once; run and retire one experiment cache.

Only newly created, private caches are managed here. Existing node caches are
never adopted or deleted. Run the *foreground worker plus audit controller*, not
a tmux/nohup launcher: its return must mean every cache consumer has exited.
This module uses only the standard library, so startup needs no NFS imports.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import time
import uuid

APPROVED = Path('/tmp/direct3d-runtime')
LIMIT = 8 * 1024**3
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.bin'}
SCHEMA = 'experiment_local_cache_v1'


class DigestReader:
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit
        self.bytes = 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.bytes += len(value)
        if self.bytes > self.limit:
            raise ValueError('Source exceeds its recorded byte budget')
        self.digest.update(value)
        return value


class PreparationFailed(ValueError):
    def __init__(self, error, cache, identity):
        super().__init__(str(error))
        self.cache, self.identity = cache, identity


@contextmanager
def allocation_lock(parent):
    parent = plain(parent)
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o022:
        raise ValueError('Cache parent must be owned and not writable by other accounts')
    fd = os.open(parent/'ALLOCATION.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError('Unowned or writable allocation lock')
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plain(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink path is not an owned cache boundary: ' + str(path))
    return path


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def signature(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def relative(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or str(p) != name
            or name == '.' or '\\' in name):
        raise ValueError('Invalid archive relative path')
    return p


def bundle(source, output, *, max_bytes=LIMIT):
    """Source is an explicitly selected dependency/input tree, not a whole env."""
    source, output = plain(source), plain(output)
    if (not source.is_dir() or output.exists() or source.is_relative_to(output)
            or output.is_relative_to(source) or not 0 < max_bytes <= LIMIT):
        raise ValueError('Require a new separate bounded bundle directory')
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Only regular source files/directories may be packed')
        if path.is_file():
            if path.suffix.lower() in WEIGHTS:
                raise ValueError('Weights are outside this small-file cache protocol')
            files.append(path)
    total = sum(p.stat().st_size for p in files)
    # Bound extraction including per-file block rounding and manifest overhead.
    allocated = sum(((p.stat().st_size + 4095)//4096)*4096 for p in files)
    if not files or max(total, allocated) + 16*1024**2 > max_bytes:
        raise ValueError('Selected input tree exceeds its explicit cache budget')
    output.mkdir(parents=True, mode=0o700)
    archive = output/'payload.tar'
    rows = []
    with tarfile.open(archive, 'x') as tar:
        for path in files:
            before = signature(path)
            name = path.relative_to(source).as_posix()
            info = tarfile.TarInfo(name)
            info.size = before[2]
            info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
            with path.open('rb') as stream:
                reader = DigestReader(stream, before[2])
                tar.addfile(info, reader)
            if signature(path) != before or reader.bytes != before[2]:
                raise ValueError('Source changed while building bundle')
            rows.append({'name': name, 'bytes': before[2], 'sha256': reader.digest.hexdigest(),
                         'mode': info.mode})
    result = {'schema': SCHEMA, 'source': str(source), 'archive': archive.name,
              'archive_sha256': sha(archive), 'files': rows,
              'archive_bytes': archive.stat().st_size,
              'logical_bytes': total, 'max_bytes': max_bytes,
              'weights_copied': False, 'source_modified': False}
    write(output/'manifest.json', result)
    return result


def stage(manifest_path, manifest_sha, parent, tag):
    if plain(parent) != plain(APPROVED):
        raise ValueError('Require the approved cache parent')
    with allocation_lock(parent):
        return _stage(manifest_path, manifest_sha, parent, tag)


def _stage(manifest_path, manifest_sha, parent, tag):
    manifest_path, parent = plain(manifest_path), plain(parent)
    manifest_bytes = manifest_path.read_bytes()
    if (parent != plain(APPROVED) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}', tag)
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha):
        raise ValueError('Require the approved root, a unique tag and pinned manifest')
    document = json.loads(manifest_bytes)
    if (document.get('schema') != SCHEMA or document.get('archive') != 'payload.tar'
            or not 0 < document['max_bytes'] <= LIMIT
            or type(document.get('archive_bytes')) is not int
            or not 0 < document['archive_bytes'] <= document['max_bytes']):
        raise ValueError('Unsupported bundle manifest')
    rows = {r['name']: r for r in document['files']}
    if len(rows) != len(document['files']) or not rows:
        raise ValueError('Duplicate or empty input manifest')
    for name, row in rows.items():
        relative(name)
        if (type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                or type(row['mode']) is not int or row['mode'] & ~0o777
                or PurePosixPath(name).suffix.lower() in WEIGHTS):
            raise ValueError('Invalid bounded manifest entry')
    expected = sum(r['bytes'] for r in rows.values())
    blocks = sum(((r['bytes']+4095)//4096)*4096 for r in rows.values())
    if expected != document['logical_bytes'] or max(expected, blocks)+16*1024**2 > document['max_bytes']:
        raise ValueError('Bundle exceeds declared extraction budget')
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o022:
        raise ValueError('Cache parent must be private and owned by this account')
    # Bound the total footprint of these small-file caches, including other runs.
    used = 0
    for root, dirs, files in os.walk(parent, followlinks=False):
        for name in [*dirs, *files]:
            path = Path(root)/name
            used += path.lstat().st_blocks*512
    if used + blocks + 16*1024**2 > LIMIT:
        raise ValueError('Existing node caches plus this payload exceed the 8GiB authorization')
    if shutil.disk_usage(parent).free < document['max_bytes']:
        raise ValueError('Insufficient local space for this bounded cache')
    cache = parent/(tag+'-'+uuid.uuid4().hex)
    cache.mkdir(mode=0o700)
    token = uuid.uuid4().hex
    created_ticks = None
    boot_id = None
    if Path('/proc/sys/kernel/random/boot_id').is_file():
        created_ticks = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf('SC_CLK_TCK'))
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identity = {'schema': SCHEMA, 'cache': str(cache), 'uid': os.getuid(),
                'token': token, 'manifest_sha256': manifest_sha,
                'device': cache.stat().st_dev, 'inode': cache.stat().st_ino,
                'created_boot_ticks': created_ticks, 'boot_id': boot_id}
    write(cache/'OWNER.json', identity)
    try:
        _extract(manifest_path, cache, document, rows)
    except Exception as error:
        raise PreparationFailed(error, cache, identity) from error
    return cache, identity


def _extract(manifest_path, cache, document, rows):
    payload = cache/'payload'; payload.mkdir()
    # Stream one sequential NFS archive read straight into local files. No
    # second full-size tar copy consumes local space alongside the payload.
    seen = set()
    with plain(manifest_path.parent/'payload.tar').open('rb') as source:
        reader = DigestReader(source, document['archive_bytes'])
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            for member in tar:
                relative(member.name)
                row = rows.get(member.name)
                if (not member.isfile() or row is None or member.name in seen
                        or member.size != row['bytes'] or member.mode != row['mode']):
                    raise ValueError('Archive contains an unplanned member/link or altered metadata')
                target = payload/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('xb') as dest:
                    shutil.copyfileobj(src, dest, 1024**2)
                if sha(target) != row['sha256']:
                    raise ValueError('Extracted input differs from source manifest')
                target.chmod(row['mode']); seen.add(member.name)
        while reader.read(1024**2):
            pass
    if reader.bytes != document['archive_bytes'] or reader.digest.hexdigest() != document['archive_sha256']:
        raise ValueError('Transferred archive checksum differs; command not started')
    if seen != set(rows):
        raise ValueError('Archive is missing planned inputs')


def group_gone(pgid):
    if pgid is None:
        return True  # only a preparation failure, before Popen was attempted
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def cache_consumers(cache, identity=None, proc=Path('/proc')):
    """Catch Linux descendants that called setsid and escaped the original PGID.

    Fresh workers and their descendants must retain the per-run environment
    marker. Pre-existing processes cannot inherit a marker for a private cache
    that did not exist when they started. Unknown processes born after this
    cache was created remain conservative blockers. Existing daemon/pool reuse
    is outside this foreground-controller protocol.
    """
    if not proc.is_dir():
        return []
    expected = ('DIRECT3D_EXPERIMENT_CACHE='+str(Path(cache)/'payload')).encode()
    created_ticks = (identity or {}).get('created_boot_ticks')
    consumers = []
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            if created_ticks is not None:
                fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[19]) < created_ticks:
                    continue
            if expected in (path/'environ').read_bytes().split(b'\0'):
                consumers.append({'pid': int(path.name), 'state': 'cache_marker_present'})
        except FileNotFoundError:
            pass  # an observed process exited during this read
        except (PermissionError, OSError, ValueError, IndexError):
            consumers.append({'pid': int(path.name), 'state': 'owned_process_identity_unreadable'})
    return consumers


def remove_owned_tree(cache, identity):
    """Anchor deletion to the opened original inode, rather than its pathname."""
    parent_fd = os.open(cache.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root_fd = None
    def clear(fd):
        with os.scandir(fd) as scan:
            for item in scan:
                if item.is_dir(follow_symlinks=False):
                    subfd = os.open(item.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        opened = os.fstat(subfd)
                        clear(subfd)
                        named = os.stat(item.name, dir_fd=fd, follow_symlinks=False)
                        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                            raise ValueError('Directory replaced while retiring cache')
                        os.rmdir(item.name, dir_fd=fd)
                    finally:
                        os.close(subfd)
                else:
                    os.unlink(item.name, dir_fd=fd)  # unlink links; never follow them
    try:
        root_fd = os.open(cache.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if (opened.st_dev, opened.st_ino) != (identity['device'], identity['inode']):
            raise ValueError('Cache inode changed before fd-based deletion')
        clear(root_fd)
        named = os.stat(cache.name, dir_fd=parent_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (identity['device'], identity['inode']):
            raise ValueError('Cache pathname replaced; preserved replacement directory')
        os.rmdir(cache.name, dir_fd=parent_fd)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def retire(cache, identity, receipt, *, pgid, required_files=(), terminal_files=()):
    """Only an owned new cache; retain on uncertain liveness or missing outputs."""
    cache, receipt = plain(cache), plain(receipt)
    if receipt == cache or receipt.is_relative_to(cache):
        raise ValueError('Cleanup receipts must survive outside the temporary cache')
    if (cache.parent != plain(APPROVED) or identity.get('schema') != SCHEMA
            or identity.get('uid') != os.getuid() or identity.get('cache') != str(cache)
            or json.loads((cache/'OWNER.json').read_text()) != identity
            or (cache.stat().st_dev, cache.stat().st_ino) != (identity['device'], identity['inode'])):
        raise ValueError('Refuse to delete an unknown, replaced or unowned directory')
    if identity.get('boot_id') is not None and identity['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Cache belongs to a previous system boot; require explicit recovery')
    proof = []
    reason = None if group_gone(pgid) else 'experiment_process_group_still_exists'
    consumers = cache_consumers(cache, identity)
    if consumers:
        reason = 'cache_consumers_still_live_or_unreadable'
    for path in [*required_files, *terminal_files]:
        path = plain(path)
        if path == cache or path.is_relative_to(cache) or path.is_relative_to(plain(APPROVED)):
            raise ValueError('Required outputs and terminal records must be durable, outside the cache root')
        if not path.is_file():
            reason = 'required_shared_output_or_exit_missing'
            continue
        before = signature(path)
        row = {'path': str(path), 'bytes': before[2], 'sha256': sha(path)}
        if path in [Path(p) for p in terminal_files]:
            try:
                if before[2] > 64:
                    raise ValueError('Oversized terminal record')
                content = path.read_bytes()
                row['sha256'] = hashlib.sha256(content).hexdigest()
                row['exit_code'] = int(content.strip())
            except ValueError:
                reason = 'terminal_exit_is_not_an_integer'
        if signature(path) != before:
            reason = 'durable_output_changed_during_verification'
        proof.append(row)
    result = {'schema': SCHEMA, 'cache': str(cache), 'identity': identity,
              'process_group': pgid, 'cache_consumers': consumers, 'durable_files': proof, 'deleted': False,
              'reason': reason, 'time_unix': time.time()}
    write(receipt/'cleanup_plan.json', result)
    if reason is None:
        if not group_gone(pgid) or cache_consumers(cache, identity):
            raise RuntimeError('Cache consumer appeared before cleanup')
        with allocation_lock(cache.parent):
            remove_owned_tree(cache, identity)
        result['deleted'] = True
    write(receipt/'cleanup.json', result)
    return result


def run(manifest, manifest_sha, parent, tag, receipt, command, *, pythonpaths=(),
        required_files=(), terminal_files=()):
    receipt = plain(receipt)
    if receipt.exists() or receipt.is_relative_to(plain(parent)) or not command:
        raise ValueError('Require new durable receipts and an explicit foreground controller')
    receipt.mkdir(parents=True, mode=0o700)
    try:
        cache, identity = stage(manifest, manifest_sha, parent, tag)
    except PreparationFailed as error:
        write(receipt/'preparation_failure.json', {'error': str(error), 'command_started': False})
        retire(error.cache, error.identity, receipt, pgid=None,
               required_files=[receipt/'preparation_failure.json'])
        raise
    env = {**os.environ, 'DIRECT3D_EXPERIMENT_CACHE': str(cache/'payload'),
           'PYTHONDONTWRITEBYTECODE': '1'}
    imports = []
    try:
        for name in pythonpaths:
            # '.' denotes the payload root; every other path is strictly relative.
            if name != '.': relative(name)
            path = cache/'payload'/name
            if not path.is_dir(): raise ValueError('Missing explicit Python path in payload')
            imports.append(str(path))
    except ValueError as error:
        write(receipt/'preparation_failure.json', {'error': str(error), 'command_started': False})
        retire(cache, identity, receipt, pgid=None, required_files=[receipt/'preparation_failure.json'])
        raise
    if imports:
        env['PYTHONPATH'] = os.pathsep.join(imports + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    write(receipt/'cache_ready.json', {'identity': identity, 'cache': str(cache),
          'command': command, 'pythonpaths': imports, 'source_modified': False})
    with (receipt/'stdout.log').open('x') as out, (receipt/'stderr.log').open('x') as err:
        try:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=True)
        except OSError as error:
            write(receipt/'start_failure.json', {'error': str(error), 'command_started': False})
            retire(cache, identity, receipt, pgid=None, required_files=[receipt/'start_failure.json'])
            raise
        write(receipt/'process.json', {'pid': child.pid, 'pgid': child.pid})
        code = child.wait()
    write(receipt/'process_exit.json', {'returncode': code, 'pid': child.pid})
    result = retire(cache, identity, receipt, pgid=child.pid,
                    required_files=[receipt/'stdout.log', receipt/'stderr.log',
                                    receipt/'process_exit.json', *required_files],
                    terminal_files=terminal_files)
    return code, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    pack = sub.add_parser('bundle')
    pack.add_argument('--source', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--max-bytes', type=int, default=LIMIT)
    execute = sub.add_parser('run')
    execute.add_argument('--manifest', type=Path, required=True)
    execute.add_argument('--manifest-sha256', required=True)
    execute.add_argument('--parent', type=Path, default=APPROVED)
    execute.add_argument('--tag', required=True)
    execute.add_argument('--receipt', type=Path, required=True)
    execute.add_argument('--pythonpath', action='append', default=[])
    execute.add_argument('--required-file', type=Path, action='append', default=[])
    execute.add_argument('--terminal-file', type=Path, action='append', default=[])
    execute.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'bundle':
        result = bundle(args.source, args.output, max_bytes=args.max_bytes)
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        code, result = run(args.manifest, args.manifest_sha256, args.parent, args.tag,
                           args.receipt, command, pythonpaths=args.pythonpath,
                           required_files=args.required_file, terminal_files=args.terminal_file)
        print(json.dumps(result), flush=True)
        raise SystemExit(code if code else (0 if result['deleted'] else 3))


if __name__ == '__main__':
    main()
