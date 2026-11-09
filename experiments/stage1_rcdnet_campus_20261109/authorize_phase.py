"""Write a fresh phase gate only through a GPU-coordinated SSH invocation."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

TASK=Path('/data1/zzy26/outputs/vistar_rcdnet_campus_20261109')
p=argparse.ArgumentParser()
p.add_argument('--phase',required=True,choices=['smoke2','whole_resize512','native_tile512'])
p.add_argument('--reservation-id',required=True)
args=p.parse_args()
uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
assert uuid.startswith('GPU-') and ',' not in uuid
assert json.loads((TASK/'AWAITING_GPU.json').read_text())['phase']==args.phase
assert json.loads((TASK/'cpu_extension_imports.terminal.json').read_text())['exit_code']==0
gate=TASK/('LAUNCH_'+args.phase+'.json')
assert not gate.exists(),'This phase was already authorized; inspect its actual terminal state'
ready=json.loads((TASK/'READY.json').read_text())
files=[p for p in (TASK/'runtime_code').rglob('*.py') if '__pycache__' not in p.parts]
vendor=Path(ready['vendor'])/'RCDNet'
files += [p for p in vendor.rglob('*') if p.is_file() and p.suffix in ('.py','.cu','.cpp','.h','.cuh','.so')
          and 'build' not in p.relative_to(vendor).parts and '__pycache__' not in p.parts]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
manifest=TASK/'runtime_manifest_frozen.json'
if manifest.exists():
    assert json.loads(manifest.read_text())==hashes,'Runtime source changed between phases'
else:
    manifest.write_text(json.dumps(hashes,indent=2)+'\n')
value=dict(phase=args.phase,reservation_id=args.reservation_id,gpu_uuid=uuid,
           created_epoch=time.time(),runtime_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
temporary=gate.with_suffix('.new')
temporary.write_text(json.dumps(value,indent=2)+'\n')
temporary.replace(gate)
print(json.dumps(value),flush=True)
