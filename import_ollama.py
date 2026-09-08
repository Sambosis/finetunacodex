"""Import BOTH language GGUF and vision projector through Ollama's files API.

A Modelfile with only FROM language.gguf silently creates a text-only model.
This importer refuses to create that incomplete model and verifies capabilities.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def request(url, path, data):
    req = urllib.request.Request(url.rstrip('/')+path, json.dumps(data).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

def upload_blob(url, path, digest):
    endpoint = url.rstrip('/')+'/api/blobs/sha256:'+digest
    try:
        with urllib.request.urlopen(urllib.request.Request(endpoint,method='HEAD'), timeout=20):
            return
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    parsed = urllib.parse.urlsplit(endpoint)
    cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=600)
    try:
        conn.putrequest('POST', parsed.path)
        conn.putheader('Content-Length',str(path.stat().st_size))
        conn.putheader('Content-Type','application/octet-stream')
        conn.endheaders()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):
                conn.send(chunk)
        response = conn.getresponse()
        body = response.read()
        if response.status not in (200,201):
            raise RuntimeError(f'Blob upload HTTP {response.status}: {body[:200]!r}')
    finally:
        conn.close()

def verified_files(folder):
    hashes = json.loads((folder/'SHA256SUMS.json').read_text())
    # Repaired legacy exports have a separate, explicit provenance manifest.
    if (folder/'VISION.json').exists():
        vision = json.loads((folder/'VISION.json').read_text())
        hashes[vision['filename']] = vision['sha256']
    model = folder/'qwen35-4b-web-tools-Q4_K_M.gguf'
    projectors = [p for p in folder.rglob('*.gguf') if 'mmproj' in p.name.lower()]
    if len(projectors) != 1:
        raise RuntimeError(f'Expected exactly one vision projector; found {len(projectors)}. Refusing text-only import.')
    files = {}
    for path in [model, projectors[0]]:
        name = path.relative_to(folder).as_posix()
        expected = hashes.get(name)
        if not expected or sha256(path) != expected:
            raise RuntimeError('Missing or mismatched SHA256: '+name)
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise RuntimeError('Not a GGUF file: '+name)
        files[name] = 'sha256:'+expected
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--model',default='qwen35-web-tools:4b')
    parser.add_argument('--url',default='http://127.0.0.1:11434')
    args = parser.parse_args()
    files = verified_files(args.artifacts)
    for name,digest in files.items():
        print('Verifying/uploading',name,flush=True)
        upload_blob(args.url,args.artifacts/name,digest.removeprefix('sha256:'))
    result = request(args.url,'/api/create',{'model':args.model,'files':files,
        'template':'{{ .Prompt }}','renderer':'qwen3.5','parser':'qwen3.5',
        'parameters':{'num_ctx':4096,'temperature':0.2,'top_p':0.95},'stream':False})
    if result.get('status') != 'success':
        raise RuntimeError('Ollama create did not succeed: '+str(result))
    info = request(args.url,'/api/show',{'model':args.model})
    caps = info.get('capabilities',[])
    if not {'vision','tools'}.issubset(caps):
        raise RuntimeError('Imported model is missing required capabilities: '+str(caps))
    (args.artifacts/'ollama-import.json').write_text(json.dumps({'model':args.model,'files':files,'capabilities':caps},indent=2))
    print('Verified capabilities:',', '.join(caps))

if __name__ == '__main__': main()
