"""Real Ollama image probes using controlled synthetic PNG fixtures, no external data."""
import argparse
import base64
import json
from pathlib import Path
import struct
import urllib.request
import zlib

def png(color, squares=False):
    width=224
    rows=[]
    for y in range(width):
        row=bytearray()
        for x in range(width):
            inside=squares and 80<=y<130 and any(a<=x<a+40 for a in [20,90,160])
            row.extend((0,0,0) if inside else color)
        rows.append(b'\0'+row)
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,width,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')

def chat(model,payload):
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',json.dumps({'model':model,'stream':False,'think':False,'keep_alive':'2m',
        'options':{'temperature':0,'num_ctx':2048,'num_predict':32,'num_thread':2,'num_gpu':0},**payload}).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:
        return json.load(response)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='qwen35-web-tools-vision-check:4b'); a=p.parse_args()
    out=Path('artifacts/vision-verification'); out.mkdir(parents=True,exist_ok=True)
    cases=[('red',(255,0,0),False,'What is the dominant color of this image? Answer with one color word.'),
           ('blue',(0,0,255),False,'What is the dominant color of this image? Answer with one color word.'),
           ('3',(255,255,255),True,'How many black squares are visible? Answer with just the number.')]
    results=[]
    try:
        for i,(expected,color,squares,prompt) in enumerate(cases):
            data=png(color,squares); (out/f'probe-{i}.png').write_bytes(data)
            response=chat(a.model,{'messages':[{'role':'user','content':prompt,'images':[base64.b64encode(data).decode()]}]})
            actual=response['message']['content'].strip().lower().rstrip('.')
            result={'expected':expected,'actual':actual,'passed':actual==expected,'eval_count':response.get('eval_count')}
            results.append(result); print(json.dumps(result),flush=True)
        tools=json.loads(Path('website-tools.json').read_text())
        r=chat(a.model,{'messages':[{'role':'user','content':'Use the calculator to evaluate 17 * 23.'}],'tools':tools})
        calls=r['message'].get('tool_calls',[])
        passed=any(c['function']['name']=='calculator' and c['function']['arguments'].get('expression','').replace(' ','')=='17*23' for c in calls)
        results.append({'tool_calls':calls,'passed':passed}); print(json.dumps(results[-1]),flush=True)
        (out/'results.json').write_text(json.dumps({'model':a.model,'results':results,'all_passed':all(r['passed'] for r in results)},indent=2))
    finally:
        req=urllib.request.Request('http://127.0.0.1:11434/api/generate',json.dumps({'model':a.model,'keep_alive':0}).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response: response.read()
    if not all(r['passed'] for r in results): raise SystemExit('One or more image/tool probes failed')

if __name__=='__main__': main()
