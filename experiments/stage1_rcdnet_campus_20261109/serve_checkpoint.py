"""Serve only the public official checkpoint to the authorized campus node."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import secrets
import time

ROOT = Path(__file__).resolve().parent
SIZE = 352523328
TOKEN = secrets.token_urlsafe(24)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/' + TOKEN + '/SECOND-model.safetensors':
            self.send_error(404)
            return
        p = ROOT / 'SECOND-model.safetensors'
        if not p.exists():
            p = ROOT / 'SECOND-model.part'
        if not p.exists():
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(SIZE))
        self.end_headers()
        sent = 0
        deadline = time.monotonic() + 1200
        with p.open('rb') as f:
            while sent < SIZE:
                b = f.read(min(1024**2, SIZE - sent))
                if not b:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Source download stalled')
                    time.sleep(0.2)
                    continue
                self.wfile.write(b)
                sent += len(b)
        print('TRANSFER COMPLETE', sent, flush=True)


server = HTTPServer(('10.135.11.120', 0), Handler)
url = f'http://10.135.11.120:{server.server_port}/{TOKEN}/SECOND-model.safetensors'
(ROOT / 'checkpoint_transfer.json').write_text(json.dumps({'url': url, 'bytes': SIZE}) + '\n')
print(json.dumps({'url': url, 'bytes': SIZE}), flush=True)
server.serve_forever()
