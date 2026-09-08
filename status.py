"""Read only this launcher's pods and download sanitized provider logs."""
import json
import os
from pathlib import Path
import urllib.request
import launch

def main():
    launch.load_env()
    json_files = sorted((launch.ROOT/'runs').glob('*.json'))
    if not json_files:
        print('No previous runs found in runs/')
        return
    state_path = json_files[-1]
    state = json.loads(state_path.read_text())
    pod_id = state.get('pod_id')
    if not pod_id:
        print('No pod ID in latest run')
        return
    pod = launch.api('GET','/pods/'+pod_id)
    print(json.dumps({k:pod.get(k) for k in ['id','name','desiredStatus','costPerHr','lastStartedAt']} if pod else {'deleted':True}))
    req = urllib.request.Request(f'https://v2-rest.runpod.io/v2/pods/{pod_id}/logs',
        headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'User-Agent':'finetunacodex/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            logs = response.read().decode(errors='replace')
        for key in ('RUNPOD_API_KEY','HF_TOKEN','WANDB_API_KEY'):
            if os.getenv(key):
                logs = logs.replace(os.environ[key],'[REDACTED]')
        (state_path.parent/(state['run_id']+'-provider.log')).write_text(logs,encoding='utf-8')
        print(logs[-8000:])
    except Exception as exc:
        print('Log fetch:',type(exc).__name__)

if __name__ == '__main__': main()
