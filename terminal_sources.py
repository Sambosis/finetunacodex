"""Fetch pinned public sources for the terminal curriculum; never execute source code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent / 'data' / 'terminal-v1'
RAW = ROOT / 'sources'

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        path.with_suffix(path.suffix + '.partial').write_bytes(content)
        path.with_suffix(path.suffix + '.partial').replace(path)
    return {'url': url, 'path': str(path.relative_to(ROOT)),
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def api(url):
    return json.load(urllib.request.urlopen(url, timeout=90))

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lockpath = ROOT / 'source-lock.json'
    if lockpath.exists():
        lock = json.loads(lockpath.read_text())
    else:
        hub = api('https://huggingface.co/api/datasets/carosh/cli-1m')
        docs = api('https://api.github.com/repos/MicrosoftDocs/PowerShell-Docs/commits/main')
        lock = {'cli_revision': hub['sha'], 'docs_revision': docs['sha']}
        lockpath.write_text(json.dumps(lock, indent=2))
    fetch('https://datasets-server.huggingface.co/splits?dataset=carosh%2Fcli-1m', RAW/'viewer-splits.json')
    hub = api('https://huggingface.co/api/datasets/carosh/cli-1m/revision/' + lock['cli_revision'])
    jobs = []
    for item in hub['siblings']:
        name = item['rfilename']
        if name.startswith('data/default/train/') or name in ['README.md', 'LICENSE']:
            jobs.append((f"https://huggingface.co/datasets/carosh/cli-1m/resolve/{lock['cli_revision']}/{name}", RAW/'cli'/name))
    jobs.append((f"https://codeload.github.com/MicrosoftDocs/PowerShell-Docs/tar.gz/{lock['docs_revision']}", RAW/'powershell-docs.tar.gz'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(lambda args: fetch(*args), jobs))
    lock['files'] = files
    # Windows administrative modules are documented in this separate official repo.
    if 'windows_docs_revision' not in lock:
        lock['windows_docs_revision'] = api('https://api.github.com/repos/MicrosoftDocs/windows-powershell-docs/commits/main')['sha']
        lockpath.write_text(json.dumps(lock, indent=2))
    rev = lock['windows_docs_revision']
    jobs = []
    for module in ['ScheduledTasks', 'NetTCPIP', 'NetAdapter', 'NetSecurity', 'DISM', 'Storage', 'Appx', 'PnpDevice']:
        path = 'docset/winserver2022-ps/' + module
        listing = api(f'https://api.github.com/repos/MicrosoftDocs/windows-powershell-docs/contents/{path}?ref={rev}')
        for item in listing:
            if item['name'].endswith('.md'):
                jobs.append((f"https://raw.githubusercontent.com/MicrosoftDocs/windows-powershell-docs/{rev}/{item['path']}", RAW/'windows-docs'/item['path']))
    for name in ['LICENSE', 'LICENSE-CODE', 'README.md']:
        jobs.append((f'https://raw.githubusercontent.com/MicrosoftDocs/windows-powershell-docs/{rev}/{name}', RAW/'windows-docs'/name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        files.extend(pool.map(lambda args: fetch(*args), jobs))
    if 'tldr_revision' not in lock:
        lock['tldr_revision'] = api('https://api.github.com/repos/tldr-pages/tldr/commits/main')['sha']
    files.append(fetch(f"https://codeload.github.com/tldr-pages/tldr/tar.gz/{lock['tldr_revision']}", RAW/'tldr.tar.gz'))
    lockpath.write_text(json.dumps(lock, indent=2))
    print(json.dumps({'revisions': {k:v for k,v in lock.items() if k!='files'}, 'files':len(files), 'bytes':sum(f['bytes'] for f in files)}, indent=2))

if __name__ == '__main__':
    main()
