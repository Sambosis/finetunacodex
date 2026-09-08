"""Download pinned public data, validate calls, split by tool set, render native Qwen.

No model weights, CUDA, API keys, or paid compute required.
"""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

import jsonschema
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
SOURCE = 'NousResearch/hermes-function-calling-v1'
REVISION = 'dae3e1d28cfbcf4b915c04ea1e072030529b4bda'
MODEL = 'Qwen/Qwen3.5-4B'
MODEL_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAX_LENGTH = 4096

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.partial')
        urllib.request.urlretrieve(url, temp)
        temp.replace(path)

def validate_calls(tools, calls):
    schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
    if len(schemas) != len(tools):
        raise ValueError('duplicate tool names')
    for call in calls:
        f = call['function']
        if f['name'] not in schemas:
            raise ValueError('unknown tool')
        if not isinstance(f['arguments'], dict):
            raise ValueError('arguments must be object')
        schema = schemas[f['name']]
        jsonschema.Draft7Validator.check_schema(schema)
        jsonschema.validate(f['arguments'], schema)
        if set(f['arguments']) - set(schema.get('properties', {})):
            raise ValueError('unknown argument')

def normalize(row):
    tools = json.loads(row['tools']) if isinstance(row['tools'], str) else row['tools']
    if not tools:
        raise ValueError('no tools')
    messages, pending = [], []
    for m in row['conversations']:
        role, value = m['from'], m['value'].strip()
        if role == 'system':
            continue  # Remove Hermes-specific serialization; Qwen supplies its own.
        if role == 'human':
            if pending:
                raise ValueError('unanswered tool calls')
            messages.append({'role': 'user', 'content': value})
        elif role == 'gpt':
            if pending:
                raise ValueError('unanswered tool calls')
            blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', value, re.S)
            content = re.sub(r'<tool_call>.*?</tool_call>', '', value, flags=re.S).strip()
            if '<tool_call' in content or '</tool_call>' in content:
                raise ValueError('malformed call tag')
            msg = {'role': 'assistant', 'content': content}
            if blocks:
                calls = []
                for block in blocks:
                    f = json.loads(block)
                    if set(f) != {'name', 'arguments'}:
                        raise ValueError('unexpected function fields')
                    calls.append({'type': 'function', 'function': f})
                validate_calls(tools, calls)
                msg['tool_calls'] = calls
                pending = [c['function']['name'] for c in calls]
            messages.append(msg)
        elif role == 'tool':
            blocks = re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>', value, re.S)
            if not blocks:
                raise ValueError('malformed response')
            for block in blocks:
                response = json.loads(block)
                name = response['name']
                if name not in pending:
                    raise ValueError('unmatched tool result')
                pending.remove(name)
                # Website sends unnamed tool messages with JSON string content.
                messages.append({'role': 'tool', 'content': json.dumps(response['content'], ensure_ascii=False)})
        else:
            raise ValueError('unexpected role')
    if not messages or messages[0]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError('incomplete conversation')
    return {'tools': tools, 'messages': messages, 'source': SOURCE, 'source_id': row['id']}

def template_renderer():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    def fail(message):
        raise ValueError(message)
    env.globals['raise_exception'] = fail
    return env.from_string((ROOT / 'research/chat_template.jinja').read_text(encoding='utf-8'))

def render_samples(row, template, tokenizer):
    """One prompt/completion per assistant turn; never train on tool/user tokens.

    Rendering both prefixes prevents masking the wrong turn in multi-turn data.
    Entire overlength samples are rejected rather than truncating call JSON/XML.
    """
    result = []
    for index, message in enumerate(row['messages']):
        if message['role'] != 'assistant':
            continue
        context = row['messages'][:index]
        prompt = template.render(messages=context, tools=row['tools'], add_generation_prompt=True, enable_thinking=False)
        full = template.render(messages=context + [message], tools=row['tools'], add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError('chat template prefix mismatch')
        ids = tokenizer.encode(full, add_special_tokens=False).ids
        if len(ids) > MAX_LENGTH:
            continue
        result.append({'prompt': prompt, 'completion': full[len(prompt):], 'tokens': len(ids),
                       'tools': row['tools'], 'messages': context, 'expected': message,
                       'source': row['source'], 'source_id': row['source_id']})
    return result

def website_rows(tools):
    """Small deterministic curriculum; these are synthetic, not benchmark evidence."""
    def row(prompt, name=None, args=None, answer=None, result=None):
        assistant = {'role': 'assistant', 'content': answer or ''}
        if name:
            assistant['tool_calls'] = [{'type': 'function', 'function': {'name': name, 'arguments': args}}]
            validate_calls(tools, assistant['tool_calls'])
        messages = [{'role': 'user', 'content': prompt}, assistant]
        if result is not None:
            messages += [{'role': 'tool', 'content': json.dumps(result)}, {'role': 'assistant', 'content': str(result['result'])}]
        return {'tools': tools, 'messages': messages, 'source': 'website-synthetic-v1', 'source_id': digest(prompt)}
    rows = []
    for a in range(11, 61):
        b = a + 7
        expr = f'{a} * {b}'
        rows.append(row(f'Calculate {a} multiplied by {b}.', 'calculator', {'expression': expr}, result={'expression': expr, 'value': a*b, 'result': str(a*b)}))
        rows.append(row(f'Use Python to print the squares of integers from 1 through {a}.', 'run_code', {'language': 'python', 'code': f'print([i*i for i in range(1, {a+1})])'}))
    for zone in ['UTC', 'America/New_York', 'Europe/London', 'Asia/Tokyo', 'Europe/Paris', 'Australia/Sydney']:
        for prefix in ['What time is it in', 'Get the current date and time in', 'Check the time right now in']:
            rows.append(row(f'{prefix} {zone}.', 'get_current_time', {'timezone': zone}))
    for topic in ['packing', 'groceries', 'project ideas', 'books', 'garden', 'travel', 'meeting', 'recipes', 'repairs', 'exercise']:
        rows += [row(f'Search my saved notes for {topic}.', 'manage_notes', {'action': 'search', 'query': topic}),
                 row(f'Create a note titled "{topic}" with content "Review this list tomorrow".', 'manage_notes', {'action': 'create', 'title': topic, 'content': 'Review this list tomorrow'}),
                 row(f'Search the web for {topic} tips.', 'web_search', {'query': f'{topic} tips'})]
    for url in ['https://docs.python.org/3/', 'https://docs.ollama.com/', 'https://en.wikipedia.org/wiki/Mars', 'https://developer.mozilla.org/en-US/docs/Web/JavaScript']:
        rows.append(row(f'Read the page at {url}', 'fetch_webpage', {'url': url}))
    for prompt, answer in [
        ('Hello!', 'Hello! How can I help?'), ('Thank you.', "You're welcome!"),
        ('Rewrite "I am late" more politely.', 'I apologize for being late.'),
        ('What is a noun?', 'A noun is a word that names a person, place, thing, or idea.'),
        ('Send an email to Alex.', 'I cannot send email with the available tools. I can help draft it.'),
        ('Delete that note.', 'Which note would you like me to delete?'),
        ('Read the article.', 'Please provide the article URL.'),
        ('Create a note.', 'What title and content should the note have?'),
        ('Calculate it.', 'What expression would you like me to calculate?'),
        ('Explain what a browser is.', 'A browser is an application for accessing and interacting with websites.'),
    ]:
        rows.append(row(prompt, answer=answer))
    return rows

def main():
    data = ROOT / 'data'
    for name in ['func-calling.json', 'glaive-function-calling-5k.json', 'README.md']:
        download(f'https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{name}', data/name)
    for name in ['chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json']:
        download(f'https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{name}', ROOT/'research'/name)
    template = template_renderer()
    tokenizer = Tokenizer.from_file(str(ROOT/'research/tokenizer.json'))
    buckets = {'train': [], 'validation': [], 'test': []}
    rejected, seen = collections.Counter(), set()
    for filename in ['func-calling.json', 'glaive-function-calling-5k.json']:
        for source_row in json.loads((data/filename).read_text(encoding='utf-8')):
            try:
                row = normalize(source_row)
                key = digest({'tools': row['tools'], 'messages': row['messages']})
                if key in seen:
                    rejected['duplicate'] += 1
                    continue
                seen.add(key)
                # Group all conversations using the same tool names in one split.
                group = sorted(t['function']['name'] for t in row['tools'])
                part = int(digest(group)[:8], 16) % 100
                split = 'test' if part < 5 else 'validation' if part < 10 else 'train'
                samples = render_samples(row, template, tokenizer)
                if not samples:
                    rejected['overlength'] += 1
                buckets[split].extend(samples)
            except (ValueError, KeyError, TypeError, jsonschema.exceptions.ValidationError, jsonschema.exceptions.SchemaError) as exc:
                rejected[type(exc).__name__ + ':' + str(exc).splitlines()[0][:90]] += 1
    tools = json.loads((ROOT/'website-tools.json').read_text())
    for row in website_rows(tools):
        buckets['train'].extend(render_samples(row, template, tokenizer))
    rng = random.Random(3407)
    # Budget-conscious pilot: sample turns, retain every website curriculum turn.
    public = [r for r in buckets['train'] if r['source'] == SOURCE]
    local = [r for r in buckets['train'] if r['source'] != SOURCE]
    rng.shuffle(public)
    buckets['train'] = public[:4000] + local
    manifest = {'dataset': SOURCE, 'revision': REVISION, 'license': 'apache-2.0',
                'base_model': MODEL, 'model_revision': MODEL_REVISION, 'max_length': MAX_LENGTH,
                'split_method': 'hash of sorted tool names; website curriculum train only',
                'rejected': dict(rejected), 'splits': {}}
    for split, rows in buckets.items():
        rng.shuffle(rows)
        path = data / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        manifest['splits'][split] = {'rows': len(rows), 'tokens': sum(r['tokens'] for r in rows),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'call_turns': sum(bool(r['expected'].get('tool_calls')) for r in rows),
            'website_turns': sum(r['source'] != SOURCE for r in rows)}
    (data/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
