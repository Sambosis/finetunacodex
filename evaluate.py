"""Shared exact-call scoring and Ollama evaluation. Never executes generated tools."""
import argparse
import json
from pathlib import Path
import re
import urllib.request

def normalized_calls(calls):
    result = []
    for call in calls:
        f = call.get('function', call)
        args = f['arguments']
        if isinstance(args, str):
            args = json.loads(args)
        result.append(json.dumps({'name': f['name'], 'arguments': args}, sort_keys=True, ensure_ascii=False))
    return sorted(result)

def parse_qwen(text, tools):
    schemas = {t['function']['name']: t['function']['parameters'].get('properties', {}) for t in tools}
    calls = []
    for name, body in re.findall(r'<function=([^>]+)>(.*?)</function>', text, re.S):
        args = {}
        for key, value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S):
            # Remove template framing only; whitespace inside strings can be meaningful.
            if value.startswith('\n'):
                value = value[1:]
            if value.endswith('\n'):
                value = value[:-1]
            typ = schemas.get(name, {}).get(key, {}).get('type')
            if typ != 'string':
                try:
                    value = json.loads(value)
                except ValueError:
                    if value in ('True', 'False', 'None'):
                        value = {'True': True, 'False': False, 'None': None}[value]
            args[key] = value
        calls.append({'function': {'name': name, 'arguments': args}})
    malformed = (text.count('<tool_call>') != text.count('</tool_call>')
                 or text.count('<function=') != len(calls)
                 or text.count('<tool_call>') != len(calls))
    return calls, malformed

def score(expected, actual, malformed=False, content=''):
    gold = expected.get('tool_calls', [])
    try:
        exact = normalized_calls(gold) == normalized_calls(actual) and not malformed
    except (ValueError, TypeError, KeyError):
        exact = False
    # Empty generations are failures, even for no-call targets.
    plain = re.sub(r'<\|[^>]+\|>|</?think>', '', content).strip()
    if not gold and not actual and not plain:
        exact = False
    return {'exact_calls': exact, 'expected_call': bool(gold), 'actual_call': bool(actual), 'malformed': malformed}

def summarize(results):
    out = {'examples': len(results), 'exact_call_accuracy': sum(r['exact_calls'] for r in results)/max(1,len(results))}
    for name, flag in [('call', True), ('no_call', False)]:
        rows = [r for r in results if r['expected_call'] == flag]
        out[name + '_examples'] = len(rows)
        out[name + '_accuracy'] = sum(r['exact_calls'] for r in rows)/len(rows) if rows else None
    out['note'] = 'Exact argument matching is conservative; no-call scoring does not grade answer quality. No tools executed.'
    return out

def ollama_chat(url, model, messages, tools):
    payload = {'model': model, 'messages': messages, 'tools': tools, 'think': False, 'stream': True,
               'keep_alive': 0, 'options': {'temperature': 0, 'seed': 3407, 'num_ctx': 4096, 'num_predict': 512}}
    req = urllib.request.Request(url.rstrip('/')+'/api/chat', json.dumps(payload).encode(), {'Content-Type':'application/json'})
    content, calls, done = '', [], False
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            chunk = json.loads(line)
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            msg = chunk.get('message', {})
            content += msg.get('content', '')
            calls.extend(msg.get('tool_calls', []))
            done |= bool(chunk.get('done'))
    if not done:
        raise RuntimeError('Ollama stream ended without done marker')
    return content, calls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--data', default='data/website-test.jsonl')
    parser.add_argument('--output', default='artifacts/ollama-evaluation.json')
    args = parser.parse_args()
    results = []
    for line in Path(args.data).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        content, calls = ollama_chat(args.url, args.model, row['messages'], row['tools'])
        result = score(row['expected'], calls, '<tool_call>' in content, content)
        result.update({'id': row.get('source_id'), 'content': content, 'calls': calls, 'expected': row['expected']})
        results.append(result)
        print(f"{len(results)}: exact={result['exact_calls']}", flush=True)
    output = {'model': args.model, 'summary': summarize(results), 'results': results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['summary'], indent=2))

if __name__ == '__main__':
    main()
