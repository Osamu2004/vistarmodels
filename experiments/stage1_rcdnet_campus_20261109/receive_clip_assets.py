"""Seed a verified offline Hugging Face snapshot through the campus LAN."""
from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request

TASK=Path('/data1/zzy26/outputs/vistar_rcdnet_campus_20261109')
HOME=Path('/data1/zzy26/cache/vistar_rcdnet_20261109/huggingface')
base=sys.argv[1]
deadline=time.time()+1800
while True:
    try:
        manifest=json.load(urllib.request.urlopen(base+'manifest.json',timeout=30));break
    except Exception:
        if time.time()>deadline:raise
        time.sleep(10)
repo=HOME/'hub/models--openai--clip-vit-base-patch32'
snapshot=repo/'snapshots'/manifest['revision'];snapshot.mkdir(parents=True,exist_ok=True)
for name,entry in manifest['files'].items():
    assert '/' not in name and '..' not in name
    path=snapshot/name
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(base+name,timeout=60) as response,path.with_suffix(path.suffix+'.receiving').open('xb') as f:
        for b in iter(lambda:response.read(4*1024**2),b''):
            h.update(b);size+=len(b);f.write(b)
    assert h.hexdigest()==entry['sha256'] and size==entry['bytes'],name
    path.with_suffix(path.suffix+'.receiving').rename(path)
    print('VERIFIED',name,size,flush=True)
(repo/'refs').mkdir(exist_ok=True);(repo/'refs/main').write_text(manifest['revision'])
(TASK/'HF_CACHE_READY.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('OFFLINE CLIP SNAPSHOT READY',flush=True)
