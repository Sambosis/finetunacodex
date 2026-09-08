"""Read-only checks; never print credentials or full API responses."""
import json
import os
import urllib.request
import urllib.error
import urllib.parse
import launch
from huggingface_hub import HfApi
launch.load_env()
try:
    print('Hugging Face authenticated:', bool(HfApi(token=os.environ['HF_TOKEN']).whoami().get('name')))
except Exception as exc:
    print('Hugging Face check failed:', type(exc).__name__)
for name, url, data in [
    ('REST', 'https://rest.runpod.io/v1/pods', None),
    ('GraphQL bearer', 'https://api.runpod.io/graphql', {'query':'{ gpuTypes { id memoryInGb securePrice } }'}),
    ('GraphQL query-key', 'https://api.runpod.io/graphql?api_key='+urllib.parse.quote(os.environ['RUNPOD_API_KEY']), {'query':'{ gpuTypes { id memoryInGb securePrice } }'}),
]:
    req=urllib.request.Request(url,json.dumps(data).encode() if data else None,
        {'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'Content-Type':'application/json','User-Agent':'finetunacodex/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            result=json.load(response)
        if name=='REST':
            print(name, 'OK; existing pod count:',len(result))
        elif result.get('errors'):
            print(name,'GraphQL errors:',str(result['errors'])[:250])
        else:
            print(name,'OK:',[g for g in result['data']['gpuTypes'] if g['id'] in ('NVIDIA A40','NVIDIA RTX A6000','NVIDIA GeForce RTX 4090')])
    except urllib.error.HTTPError as exc:
        body=exc.read().decode(errors='replace')[:400]
        body=body.replace(os.environ['RUNPOD_API_KEY'],'[REDACTED]')
        print(name,'HTTP',exc.code,body)
    except Exception as exc:
        print(name,type(exc).__name__)
