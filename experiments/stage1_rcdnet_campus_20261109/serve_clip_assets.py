"""Temporary single-model artifact transport; no directory browsing."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import hashlib
import json
import secrets

ROOT = Path(__file__).resolve().parent / 'clip_assets'
NAMES = ('config.json','merges.txt','vocab.json','tokenizer.json',
         'tokenizer_config.json','special_tokens_map.json','pytorch_model.bin')
TOKEN = secrets.token_urlsafe(24)
manifest = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global manifest
        prefix = '/' + TOKEN + '/'
        if not self.path.startswith(prefix):
            self.send_error(404); return
        name = self.path[len(prefix):]
        if name != 'manifest.json' and name not in NAMES:
            self.send_error(404); return
        if not (ROOT/'acquisition.json').exists():
            self.send_error(503); return
        if manifest is None:
            files = {}
            for item in NAMES:
                h=hashlib.sha256()
                with (ROOT/item).open('rb') as f:
                    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
                files[item]={'bytes':(ROOT/item).stat().st_size,'sha256':h.hexdigest()}
            manifest={'revision':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','files':files}
        data = json.dumps(manifest).encode() if name == 'manifest.json' else None
        self.send_response(200)
        self.send_header('Content-Length',str(len(data) if data else (ROOT/name).stat().st_size))
        self.end_headers()
        if data:
            self.wfile.write(data)
        else:
            with (ROOT/name).open('rb') as f:
                for b in iter(lambda:f.read(4*1024**2),b''):self.wfile.write(b)
        print('SERVED',name,flush=True)


server=HTTPServer(('10.135.11.120',0),Handler)
url=f'http://10.135.11.120:{server.server_port}/{TOKEN}/'
(ROOT.parent/'clip_transfer.json').write_text(json.dumps({'base_url':url})+'\n')
print(json.dumps({'base_url':url}),flush=True)
server.serve_forever()
