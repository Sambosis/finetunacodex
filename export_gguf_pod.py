"""Rent a lightweight 1-GPU pod on RunPod to convert merged HuggingFace weights to Q4_K_M GGUF in a dedicated repo."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from huggingface_hub import HfApi
import launch

ROOT = launch.ROOT
IMAGE = launch.IMAGE

WORKER_SCRIPT = """
import os, sys, json, hashlib, time, subprocess, traceback
from pathlib import Path

source_repo = os.environ['SOURCE_REPO']
target_repo = os.environ['TARGET_REPO']
token = os.environ['HF_TOKEN']
out_dir = Path('/workspace/artifacts')
out_dir.mkdir(parents=True, exist_ok=True)
api = None

try:
    print(">>> Step 1/4: Installing unsloth and dependencies with uv...", flush=True)
    subprocess.run(['python', '-m', 'pip', 'install', 'uv'], check=True)
    subprocess.run(['python', '-m', 'uv', 'pip', 'install', '--system', '--break-system-packages', 'unsloth', 'huggingface_hub', 'hf_transfer', 'gguf', 'protobuf'], check=True)

    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    from unsloth import FastLanguageModel
    from transformers import AutoProcessor
    from vision_export import assert_vision_present, finish_vision_export, BASE, BASE_REVISION
    api = HfApi(token=token)

    print(">>> Step 2/4: Downloading merged model weights from Hugging Face...", flush=True)
    local_merged = snapshot_download(
        repo_id=source_repo,
        allow_patterns=['artifacts/merged/*'],
        token=token
    )
    model_path = Path(local_merged) / 'artifacts' / 'merged'
    print(f"Downloaded merged weights to {model_path}", flush=True)

    print(">>> Step 3/4: Loading model and exporting to Q4_K_M GGUF...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    processor = tokenizer if hasattr(tokenizer, 'image_processor') else AutoProcessor.from_pretrained(BASE, revision=BASE_REVISION)
    assert_vision_present(model, processor)

    gguf_dir = out_dir / 'gguf'
    model.save_pretrained_gguf(str(gguf_dir), processor, quantization_method='q4_k_m')

    files = [p for p in out_dir.rglob('*[Qq]4*[Mm]*.gguf') if 'mmproj' not in p.name.lower()]
    if len(files) != 1:
        raise RuntimeError(f"Could not find generated Q4_K_M GGUF file in {out_dir}")
    target_gguf = out_dir / 'qwen35-4b-web-tools-Q4_K_M.gguf'
    if files[0] != target_gguf:
        files[0].replace(target_gguf)
    finish_vision_export(out_dir, BASE, BASE_REVISION)
    print(f"Generated GGUF: {target_gguf} ({target_gguf.stat().st_size / (1024*1024*1024):.2f} GB)", flush=True)

    # Use the files API importer; do not emit a misleading text-only Modelfile.

    # Copy metric files from source repo if present
    for metric_file in ['baseline.json', 'finetuned.json', 'comparison.json', 'training.json']:
        try:
            m_path = hf_hub_download(repo_id=source_repo, filename=f'artifacts/{metric_file}', token=token)
            dest = out_dir / metric_file
            dest.write_bytes(Path(m_path).read_bytes())
        except Exception:
            pass

    # Calculate hashes
    hashes = {}
    for path in out_dir.rglob('*'):
        if path.is_file() and path.name != 'SHA256SUMS.json':
            with path.open('rb') as h:
                hashes[path.relative_to(out_dir).as_posix()] = hashlib.file_digest(h, 'sha256').hexdigest()
    (out_dir / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2))
    with (out_dir / 'SHA256SUMS.json').open('rb') as h:
        hashes['SHA256SUMS.json'] = hashlib.file_digest(h, 'sha256').hexdigest()

    print(">>> Step 4/4: Uploading GGUF artifacts and SUCCESS marker to Hugging Face...", flush=True)
    api.upload_folder(
        repo_id=target_repo,
        folder_path=str(out_dir),
        path_in_repo='artifacts',
        allow_patterns=['*.gguf', 'Modelfile', 'SHA256SUMS.json', '*.json']
    )

    success_data = {'status': 'complete', 'sha256': hashes, 'finished_at': time.time()}
    api.upload_file(repo_id=target_repo, path_in_repo='SUCCESS.json', path_or_fileobj=json.dumps(success_data).encode())
    print(">>> GGUF export and upload complete!", flush=True)
except Exception as e:
    err_text = traceback.format_exc()
    print("WORKER ERROR:", err_text, flush=True)
    if api:
        try:
            api.upload_file(repo_id=target_repo, path_in_repo='export_error.log', path_or_fileobj=err_text.encode())
        except Exception:
            pass
    raise
"""

def main():
    launch.load_env()
    source_repo = 'Sambosis/Qwen3.5-4B-PerceptionBench-LoRA-20260907-163552-8ab4bc'
    hf = HfApi(token=os.environ['HF_TOKEN'])
    user = hf.whoami()['name']
    target_repo = f'{user}/qwen35-4b-web-tools-gguf-{time.strftime("%Y%m%d-%H%M%S")}'
    
    hf.create_repo(target_repo, private=True, exist_ok=False)
    print(f"Source repo (merged weights): {source_repo}")
    print(f"Target repo (GGUF model):    {target_repo}")

    candidates = launch.gpu_prices()
    pod = None
    vision_source = base64.b64encode((ROOT/'vision_export.py').read_bytes()).decode()
    bootstrap = f"import base64,os,pathlib; pathlib.Path('/tmp/vision_export.py').write_bytes(base64.b64decode('{vision_source}')); p=pathlib.Path('/tmp/worker.py'); p.write_bytes(base64.b64decode('{base64.b64encode(WORKER_SCRIPT.encode()).decode()}')); os.execv('/usr/bin/env',['env','python','-u',str(p)])"
    
    for gpu in candidates:
        payload = {
            'name': 'qwen-gguf-exporter',
            'imageName': IMAGE,
            'cloudType': 'SECURE',
            'computeType': 'GPU',
            'gpuTypeIds': [gpu['id']],
            'gpuCount': 1,
            'interruptible': False,
            'containerDiskInGb': 80,
            'volumeInGb': 0,
            'minRAMPerGPU': 24,
            'allowedCudaVersions': ['12.8', '12.9', '13.0'],
            'ports': [],
            'dockerEntrypoint': ['python', '-u', '-c'],
            'dockerStartCmd': [bootstrap],
            'env': {
                'SOURCE_REPO': source_repo,
                'TARGET_REPO': target_repo,
                'HF_TOKEN': os.environ['HF_TOKEN'],
                'RUNPOD_API_KEY': os.environ['RUNPOD_API_KEY'],
                'HF_HUB_ENABLE_HF_TRANSFER': '1',
            }
        }
        print(f"Attempting rental on {gpu['id']} (${float(gpu['securePrice']):.2f}/hr)...", flush=True)
        try:
            pod = launch.api('POST', '/pods', payload)
            break
        except Exception as e:
            err = str(e)
            if 'no instances currently available' in err.lower() or 'unavailable' in err.lower():
                print(f"{gpu['id']} out of stock, trying next GPU...", flush=True)
                continue
            raise RuntimeError(f"Could not create pod: {err}")

    if not pod:
        raise RuntimeError("No eligible GPU instances available on RunPod right now.")

    pod_id = pod['id']
    print(f"Pod created successfully! Pod ID: {pod_id}", flush=True)
    start_time = time.time()
    success = False
    last_log_tail = ''

    try:
        # Stream status and check for completion (max 25 minutes)
        while time.time() - start_time < 1500:
            time.sleep(15)
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            try:
                tree = [f.path for f in hf.list_repo_tree(target_repo, recursive=False)]
                if 'SUCCESS.json' in tree:
                    print("\nSUCCESS.json detected in Hugging Face repository!", flush=True)
                    success = True
                    break
                if 'export_error.log' in tree:
                    err_log = hf.hf_hub_download(target_repo, 'export_error.log', token=os.environ['HF_TOKEN'])
                    print("\nWorker reported an error:\n", Path(err_log).read_text(), flush=True)
                    break
            except Exception:
                pass
            
            status_line = f"[{mins:02d}m{secs:02d}s] Pod {pod_id} converting & uploading GGUF..."
            try:
                req = urllib.request.Request(f'https://v2-rest.runpod.io/v2/pods/{pod_id}/logs',
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
    finally:
        print(f"Terminating pod {pod_id}...", flush=True)
        launch.terminate(pod_id)
        print("Pod terminated successfully.", flush=True)

    if not success:
        raise RuntimeError("GGUF conversion job did not complete successfully.")

    # Download verified result
    out_local = ROOT / 'artifacts' / '163552-8ab4bc'
    print(f"Downloading verified artifacts to {out_local}...", flush=True)
    launch.download_result(target_repo, out_local)
    print("\nSUCCESS! Artifacts downloaded and verified.")
    print("Auto-importing into Ollama now...", flush=True)
    import_script = ROOT / 'Import-Ollama.ps1'
    if import_script.exists():
        subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', str(import_script), '-ArtifactDirectory', str(out_local / 'artifacts')], check=True)
        print("Model imported and evaluated in Ollama successfully!")

if __name__ == '__main__':
    main()
