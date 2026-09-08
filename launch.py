"""RunPod one-shot launcher. Default is a free local preflight; --run rents one pod.

No blind create retries. Persistent ledger charges previous attempts conservatively.
API/provider outages can defeat application-level timers: retain billing headroom.
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
BUDGET = 14.0
RESERVE = 4.0
MAX_GPU_RATE = 1.0
DISK_GB = 100
STORAGE_RATE = DISK_GB * 0.10 / (30*24)
IMAGE = 'runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404'

def load_env():
    path = ROOT/'.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            name, sep, value = line.partition('=')
            if sep and name.strip() in ('RUNPOD_API_KEY','HF_TOKEN','HF_REPO_ID','WANDB_API_KEY','WANDB_PROJECT'):
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))

def api(method, route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request('https://rest.runpod.io/v1'+route, data,
        {'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'], 'Content-Type':'application/json', 'User-Agent':'finetunacodex/1.0'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and method in ('GET','DELETE'):
            return None
        detail = exc.read().decode(errors='replace')
        for secret_name in ('RUNPOD_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY'):
            if os.getenv(secret_name):
                detail = detail.replace(os.environ[secret_name], '[REDACTED]')
        raise RuntimeError(f'RunPod {method} {route}: HTTP {exc.code}: {detail[:600]}') from None

def gpu_prices():
    query = '{ gpuTypes { id memoryInGb securePrice } }'
    req = urllib.request.Request('https://api.runpod.io/graphql', json.dumps({'query':query}).encode(),
        {'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'], 'Content-Type':'application/json', 'User-Agent':'finetunacodex/1.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.load(response)
    if data.get('errors'):
        raise RuntimeError('GPU price query failed; refusing to guess a rental price')
    allowed = {'NVIDIA GeForce RTX 4090', 'NVIDIA A40', 'NVIDIA RTX A6000'}
    candidates = [g for g in data['data']['gpuTypes'] if g['id'] in allowed
                  and g['securePrice'] and 0 < float(g['securePrice']) <= MAX_GPU_RATE
                  and int(g['memoryInGb']) >= 24]
    if not candidates:
        raise RuntimeError('No eligible GPU at or below $1/hour')
    return sorted(candidates, key=lambda g: float(g['securePrice']))

def deadline_seconds(rate, spent=0):
    if not math.isfinite(rate) or rate <= 0 or rate > MAX_GPU_RATE:
        raise ValueError('GPU rate outside allowed range')
    remaining = BUDGET - RESERVE - spent
    if remaining <= 0:
        raise ValueError('No experiment budget remains')
    seconds = min(8*3600, int(remaining/(rate+STORAGE_RATE)*3600))
    if seconds < 3*3600:
        raise ValueError('Budget does not allow setup, training, and export')
    return seconds

def write_state(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)

def terminate(pod_id):
    for attempt in range(12):
        try:
            api('DELETE', '/pods/'+pod_id)
            if api('GET', '/pods/'+pod_id) is None:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def monitor(state_path):
    state = json.loads(state_path.read_text())
    pod_id = state['pod_id']
    while time.time() < state['deadline']:
        try:
            pod = api('GET', '/pods/'+pod_id)
            if pod is None:
                state['terminated'] = True
                state['ended_at'] = time.time()
                write_state(state_path, state)
                return
        except Exception:
            pass
        time.sleep(20)
    state['terminated'] = terminate(pod_id)
    state['ended_at'] = time.time()
    write_state(state_path, state)
    if not state['terminated']:
        raise RuntimeError('Could not verify pod deletion; open RunPod console immediately: '+pod_id)

def preflight():
    required = ['data/train.jsonl','data/validation.jsonl','data/test.jsonl','data/website-test.jsonl',
                'data/manifest.json','research/chat_template.jinja','website-tools.json','train.py','evaluate.py',
                'artifact_store.py','vision_export.py','requirements-gpu.txt','research/requirements-gpu-resolved.txt']
    for name in required:
        if not (ROOT/name).is_file():
            raise RuntimeError('Missing '+name+'; run prepare_data.py and make_website_tests.py')
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    for split, info in manifest['splits'].items():
        actual = hashlib.sha256((ROOT/'data'/f'{split}.jsonl').read_bytes()).hexdigest()
        if actual != info['sha256']:
            raise RuntimeError('Prepared dataset changed: '+split)
    print(json.dumps({'budget_usd': BUDGET, 'unallocated_reserve_usd': RESERVE,
                      'max_gpu_hourly_usd': MAX_GPU_RATE, 'maximum_hours': 8,
                      'train_examples': manifest['splits']['train']['rows'],
                      'credentials_present': {k: bool(os.getenv(k)) for k in ['RUNPOD_API_KEY','HF_TOKEN','WANDB_API_KEY']}}, indent=2))
    return required

def download_result(repo, output):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    hf = HfApi(token=os.environ['HF_TOKEN'])
    try:
        marker = hf_hub_download(repo, 'SUCCESS.json', token=os.environ['HF_TOKEN'])
    except Exception:
        raise RuntimeError('No verified SUCCESS marker; inspect private repo logs/recovery adapters. Training not complete.') from None
    success = json.loads(Path(marker).read_text())
    snapshot_download(repo, local_dir=output, token=os.environ['HF_TOKEN'],
        allow_patterns=['artifacts/*.gguf','artifacts/Modelfile','artifacts/*.json','artifacts/*.txt','SUCCESS.json','job.log'])
    for name, sha in success['sha256'].items():
        path = output/'artifacts'/name
        if path.is_file():
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != sha:
                    raise RuntimeError('Downloaded checksum failed: '+name)
    print('Verified local artifacts:', output/'artifacts')

def main():
    os.chdir(ROOT)
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--watch', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--download', help='Download an existing private HF run repo without renting a GPU')
    args = parser.parse_args()
    if args.watch:
        monitor(args.watch)
        return
    if args.download:
        download_result(args.download, ROOT/'artifacts'/args.download.split('/')[-1])
        return
    files = preflight()
    if not args.run:
        print('Preflight only. No paid resources created.')
        return
    if not os.getenv('RUNPOD_API_KEY') or not os.getenv('HF_TOKEN'):
        raise RuntimeError('Add RUNPOD_API_KEY and HF_TOKEN to the local .env file first')
    from huggingface_hub import HfApi
    runs = ROOT/'runs'
    runs.mkdir(exist_ok=True)
    # Exclusive lock also blocks two launchers from spending simultaneously.
    lock = runs/'launch.lock'
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        states = [json.loads(p.read_text()) for p in runs.glob('*.json')]
        if any(not s.get('terminated') for s in states):
            raise RuntimeError('Previous pod not confirmed terminated; resolve it before launching again')
        spent = sum(s['reserved_cost'] for s in states)
        candidates = gpu_prices()
        hf = HfApi(token=os.environ['HF_TOKEN'])
        user = hf.whoami()['name']
        run_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        # Treat configured repo as a naming prefix, never overwrite an existing adapter.
        repo = (os.getenv('HF_REPO_ID') or user+'/qwen35-4b-web-tools')+'-'+run_id
        hf.create_repo(repo, private=True, exist_ok=False)
        if not hf.model_info(repo).private:
            raise RuntimeError('Output repository must be private')
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name in files:
                bundle.write(ROOT/name, name)
            bundle.write(ROOT/'data/README.md', 'data/SOURCE_README.md')
        raw = archive.getvalue()
        commit = hf.upload_file(repo_id=repo, path_in_repo='input.zip', path_or_fileobj=raw)
        supervisor = base64.b64encode((ROOT/'supervisor.py').read_bytes()).decode()
        bootstrap = "import base64,os,pathlib; p=pathlib.Path('/tmp/finetune_supervisor.py'); p.write_bytes(base64.b64decode(os.environ.pop('BOOTSTRAP'))); os.execv('/usr/bin/env',['env','python','-u',str(p)])"
        pod = None
        state_path = None
        state = None
        for gpu in candidates:
            try:
                seconds = deadline_seconds(float(gpu['securePrice']), spent)
            except ValueError:
                continue
            start = time.time()
            deadline = start + seconds
            reserved_cost = seconds / 3600 * (float(gpu['securePrice']) + STORAGE_RATE)
            state = {'run_id': run_id, 'repo': repo, 'started_at': start, 'deadline': deadline,
                     'reserved_cost': reserved_cost, 'gpu': gpu['id'], 'terminated': False}
            state_path = runs / (run_id + '.json')
            write_state(state_path, state)
            env = {'BOOTSTRAP': supervisor, 'RUNPOD_API_KEY': os.environ['RUNPOD_API_KEY'],
                'SUPERVISOR_RUNPOD_API_KEY': os.environ['RUNPOD_API_KEY'],
                'HF_TOKEN': os.environ['HF_TOKEN'], 'HF_REPO_ID': repo, 'POD_DEADLINE': str(deadline),
                'TRAIN_DEADLINE': str(deadline-5400), 'BUNDLE_SHA256': hashlib.sha256(raw).hexdigest(),
                'BUNDLE_URL': f'https://huggingface.co/{repo}/resolve/{commit.oid}/input.zip'}
            if os.getenv('WANDB_API_KEY'):
                env['WANDB_API_KEY'] = os.environ['WANDB_API_KEY']
            if os.getenv('WANDB_PROJECT'):
                env['WANDB_PROJECT'] = os.environ['WANDB_PROJECT']
            payload = {'name': 'qwen-tools-'+run_id, 'imageName': IMAGE, 'cloudType':'SECURE',
                'computeType':'GPU', 'gpuTypeIds':[gpu['id']], 'gpuCount':1, 'interruptible':False,
                'containerDiskInGb':DISK_GB, 'volumeInGb':0, 'minRAMPerGPU':24,
                'allowedCudaVersions':['12.8','12.9','13.0'], 'ports':[],
                'dockerEntrypoint':['python','-u','-c'], 'dockerStartCmd':[bootstrap], 'env':env}
            print(f"Attempting {gpu['id']} (${float(gpu['securePrice']):.2f}/hr), {seconds/3600:.2f} hr deadline; reserved ${reserved_cost:.2f} plus ${RESERVE:.2f} headroom...", flush=True)
            try:
                pod = api('POST','/pods',payload)
                break
            except Exception as create_error:
                err_msg = str(create_error)
                matches = [p for p in (api('GET','/pods') or []) if p.get('name') == payload['name']]
                if len(matches) == 1:
                    pod = matches[0]
                    break
                elif len(matches) == 0 and ('no instances currently available' in err_msg.lower() or 'unavailable' in err_msg.lower()):
                    print(f"{gpu['id']} has no instances currently available on RunPod. Trying next eligible GPU...", flush=True)
                    continue
                else:
                    raise RuntimeError('Creation outcome unknown. Check named pod in RunPod console; do not rerun. '+err_msg) from None

        if pod is None:
            if state_path and state_path.exists():
                state_path.unlink()
            raise RuntimeError('All eligible GPUs currently have no available instances on RunPod. Please try again in a few minutes.')

        state['pod_id'] = pod['id']
        write_state(state_path, state)
        try:
            rate = max(float(pod.get('costPerHr') or 0), float(pod.get('adjustedCostPerHr') or 0))
            if not math.isfinite(rate) or rate <= 0 or rate > MAX_GPU_RATE or rate > float(gpu['securePrice'])+0.001:
                raise RuntimeError('Actual pod price not within verified quote')
            # Separate local process survives Ctrl+C in the main launcher.
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session':True}
            with (runs/(run_id+'-watch.log')).open('w') as log:
                watcher = subprocess.Popen([sys.executable,str(ROOT/'launch.py'),'--watch',str(state_path)], stdout=log, stderr=log, **kwargs)
            print('Pod:', pod['id'], '| Private artifacts:', repo, flush=True)
            if os.getenv('WANDB_API_KEY'):
                print('WandB monitoring: active (project: ' + os.getenv('WANDB_PROJECT', 'qwen35-web-tools') + ')', flush=True)
            last_log_tail = ''
            while watcher.poll() is None:
                time.sleep(25)
                elapsed = int(time.time() - start)
                mins, secs = divmod(elapsed, 60)
                try:
                    pod_info = api('GET', '/pods/'+pod['id'])
                    desired = pod_info.get('desiredStatus', 'UNKNOWN') if pod_info else 'DELETED'
                except Exception:
                    desired = 'UNKNOWN'
                status_line = f"[{mins:02d}m{secs:02d}s] Pod {pod['id']} ({desired}) - running"
                try:
                    req = urllib.request.Request(f'https://v2-rest.runpod.io/v2/pods/{pod["id"]}/logs',
                        headers={'Authorization': 'Bearer '+os.environ['RUNPOD_API_KEY'], 'User-Agent': 'finetunacodex/1.0'})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        raw = resp.read().decode(errors='replace')
                    for sec in ('RUNPOD_API_KEY', 'HF_TOKEN', 'WANDB_API_KEY'):
                        if os.getenv(sec):
                            raw = raw.replace(os.environ[sec], '[REDACTED]')
                    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
                    if lines and lines[-1] != last_log_tail:
                        last_log_tail = lines[-1]
                        status_line += f" | {last_log_tail[:100]}"
                except Exception:
                    pass
                print(status_line, flush=True)
            if watcher.returncode:
                raise RuntimeError('Local watchdog reported a failure; check RunPod console')
        finally:
            state = json.loads(state_path.read_text())
            state['terminated'] = terminate(pod['id'])
            state['ended_at'] = time.time()
            write_state(state_path,state)
            if not state['terminated']:
                raise RuntimeError('POD DELETION UNCONFIRMED: '+pod['id'])
        download_result(repo, ROOT/'artifacts'/run_id)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, FileExistsError) as exc:
        print('ERROR:', str(exc), file=sys.stderr)
        sys.exit(1)
