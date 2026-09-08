"""Runs inside the pod using stdlib before installing training dependencies.

The separate watchdog uses an absolute deadline including container startup time.
Artifacts are uploaded by train.py; timeout/failure still terminates to protect cost.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

def pod_request(method):
    pod = os.environ.get('RUNPOD_POD_ID')
    if not pod:
        return None
    api_key = os.environ.get('SUPERVISOR_RUNPOD_API_KEY') or os.environ.get('RUNPOD_API_KEY')
    if not api_key:
        return None
    req = urllib.request.Request('https://rest.runpod.io/v1/pods/'+pod,
        headers={'Authorization': 'Bearer '+api_key, 'User-Agent':'Mozilla/5.0 (compatible; RunPodSupervisor/1.0)'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except Exception:
        return None

def terminate():
    pod = os.environ.get('RUNPOD_POD_ID')
    if pod:
        try:
            subprocess.run(['runpodctl', 'pod', 'delete', pod], timeout=15)
            return
        except Exception:
            pass
        try:
            subprocess.run(['runpodctl', 'pod', 'stop', pod], timeout=15)
            return
        except Exception:
            pass
    for attempt in range(12):
        try:
            pod_request('DELETE')
            if pod_request('GET') is None:
                return
        except Exception:
            pass
        time.sleep(5)

def watchdog():
    deadline = float(os.environ['POD_DEADLINE'])
    while time.time() < deadline:
        time.sleep(min(10, max(0, deadline-time.time())))
    terminate()

def safe_extract(raw, root):
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for name in bundle.namelist():
            path = (root/name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Invalid bundle path')
        bundle.extractall(root)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'watchdog':
        watchdog()
        return
    root = Path('/workspace/job')
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    subprocess.Popen([sys.executable, __file__, 'watchdog'], start_new_session=True)
    exit_code = 1
    try:
        if time.time() >= float(os.environ['POD_DEADLINE']) - 7200:
            raise RuntimeError('Not enough runtime remaining to start safely')
        req = urllib.request.Request(os.environ['BUNDLE_URL'],
            headers={'Authorization': 'Bearer '+os.environ['HF_TOKEN']})
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != os.environ['BUNDLE_SHA256']:
            raise RuntimeError('Bundle checksum mismatch')
        safe_extract(raw, root)
        env = os.environ.copy()
        env.pop('RUNPOD_API_KEY', None)  # Trainer does not need account management access.
        env.pop('SUPERVISOR_RUNPOD_API_KEY', None)
        env.pop('BOOTSTRAP', None)
        commands = [
            ['python', '-m', 'pip', 'install', 'uv'],
            ['python', '-m', 'uv', 'pip', 'install', '--system', '--break-system-packages', '-r', 'research/requirements-gpu-resolved.txt', 'wandb', 'hf_transfer'],
            ['python', '-u', 'train.py'],
        ]
        with (root/'job.log').open('w', encoding='utf-8') as log:
            for command in commands:
                header = f">>> Running: {' '.join(command)}\n"
                sys.stdout.write(header)
                sys.stdout.flush()
                log.write(header)
                log.flush()
                process = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        log.write(line)
                        log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError('Worker phase failed with exit code '+str(code))
        exit_code = 0
    except Exception as exc:
        # Do not print HTTP bodies or environment values.
        print('Job failed:', type(exc).__name__, str(exc)[:160], flush=True)
    finally:
        # Best effort log/error backup with a finite timeout; watchdog remains active.
        env = os.environ.copy()
        env.pop('RUNPOD_API_KEY', None)
        env.pop('SUPERVISOR_RUNPOD_API_KEY', None)
        backup_code = """
import os,json,pathlib
from huggingface_hub import HfApi
api=HfApi(token=os.environ['HF_TOKEN'])
repo=os.environ['HF_REPO_ID']
if not api.model_info(repo).private: raise RuntimeError('Repo must be private')
path=pathlib.Path('/workspace/job/job.log')
if path.exists(): api.upload_file(repo_id=repo,path_or_fileobj=str(path),path_in_repo='job.log')
"""
        try:
            subprocess.run([sys.executable,'-c',backup_code], env=env, timeout=120, capture_output=True)
        except Exception:
            pass
        terminate()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
