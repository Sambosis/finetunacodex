"""Offline release checks for terminal-v1. Never execute training commands."""
import collections
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from evaluate import parse_qwen, normalized_calls
from prepare_data import ROOT, validate_calls, template_renderer, canonical
from prepare_terminal_data import OUT, readjsonl, normalized

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    tokenizer=Tokenizer.from_file(str(ROOT/'research/tokenizer.json'))
    renderer=template_renderer()
    groups={}; commands={}; ids=set(); checks=collections.Counter()
    original={canonical({k:r[k] for k in ['source','source_id','prompt','completion','expected']}) for r in readjsonl(ROOT/'data/train.jsonl')}
    for split,stats in manifest['splits'].items():
        path=OUT/(split+'.jsonl'); rows=readjsonl(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==stats['sha256']
        assert len(rows)==stats['rows']
        assert sum(r['os']=='windows' for r in rows)/len(rows)>=.5
        assert sum(r['supervised_tokens'] for r in rows if r['os']=='windows')/sum(r['supervised_tokens'] for r in rows)>=.5
        for row in rows:
            expected=row['expected'].get('tool_calls',[])
            validate_calls(row['tools'],expected)
            actual,bad=parse_qwen(row['completion'],row['tools'])
            assert not bad, row['source_id']
            assert normalized_calls(actual)==normalized_calls(expected),row['source_id']
            full=renderer.render(messages=row['messages']+[row['expected']],tools=row['tools'],add_generation_prompt=False,enable_thinking=False)
            prompt=renderer.render(messages=row['messages'],tools=row['tools'],add_generation_prompt=True,enable_thinking=False)
            assert prompt==row['prompt'] and full==prompt+row['completion']
            prefix=tokenizer.encode(prompt,add_special_tokens=False).ids
            total=tokenizer.encode(full,add_special_tokens=False).ids
            assert total[:len(prefix)]==prefix and len(total)<=4096 and len(total)>len(prefix)
            assert len(total)==row['tokens']
            if row['os']=='retention':
                assert split=='train'
                assert canonical({k:row[k] for k in ['source','source_id','prompt','completion','expected']}) in original
                checks['unchanged_original_training_replay']+=1
            else:
                assert row['id'] not in ids; ids.add(row['id'])
                assert all(m['role'] in ['system','user'] for m in row['messages'])
                assert expected[0]['function']['name']=='run_terminal'
                args=expected[0]['function']['arguments']
                assert args['shell']==row['shell']
                group=row['group_id']; cmd=normalized(args['command'])
                assert groups.setdefault(group,split)==split
                assert commands.setdefault(cmd,split)==split
                assert row['source_url'] and row['revision'] and row['license_spdx']
                checks['attributed_terminal_examples']+=1
            checks['schema_template_roundtrip_and_masking']+=1
    lock=json.loads((OUT/'source-lock.json').read_text())
    for file in lock['files']:
        path=OUT/file['path']
        assert path.stat().st_size==file['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
        checks['source_file_hashes']+=1
    result={'passed':True,'checks':dict(checks),'split_group_overlap':0,'normalized_command_overlap_between_splits':0,'windows_majority_by_rows_and_supervised_tokens':True,'execution_testing':False}
    (OUT/'validation-report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
