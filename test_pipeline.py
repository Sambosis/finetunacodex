import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import hashlib
import os
from types import SimpleNamespace

import launch
from evaluate import parse_qwen, score, normalized_calls
from prepare_data import ROOT, normalize, template_renderer, render_samples, validate_calls
from supervisor import safe_extract
from tokenizers import Tokenizer
from artifact_store import upload_verified

class PipelineTests(unittest.TestCase):
    def test_artifact_hashes_and_private_requirement(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{'HF_TOKEN':'test','HF_REPO_ID':'test/private'}):
            folder=Path(temp)
            (folder/'model.gguf').write_bytes(b'GGUF test')
            entry=SimpleNamespace(path='artifacts/model.gguf',size=9,lfs=SimpleNamespace(sha256=hashlib.sha256(b'GGUF test').hexdigest()))
            with patch('artifact_store.HfApi') as factory:
                api=factory.return_value
                api.model_info.return_value.private=True
                api.list_repo_tree.return_value=[entry]
                upload_verified(folder,'artifacts')
                entry.lfs.sha256='bad'
                with self.assertRaises(RuntimeError): upload_verified(folder,'artifacts')
                api.model_info.return_value.private=False
                with self.assertRaises(RuntimeError): upload_verified(folder,'artifacts')

    def test_launcher_lifecycle_without_paid_api(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{'RUNPOD_API_KEY':'test-key','HF_TOKEN':'test-token'},clear=True):
            root=Path(temp)
            (root/'data').mkdir(); (root/'data/README.md').write_text('dataset license')
            (root/'supervisor.py').write_text('print("fixture")')
            (root/'fixture').write_text('fixture')
            previous=os.getcwd()
            try:
                with patch('launch.ROOT',root),patch('launch.preflight',return_value=['fixture']),patch('launch.gpu_prices',return_value=[{'id':'NVIDIA A40','securePrice':0.5}]),patch('huggingface_hub.HfApi') as hf_factory,patch('launch.api') as api,patch('launch.subprocess.Popen') as popen,patch('launch.download_result') as download,patch('sys.argv',['launch.py','--run']):
                    hf=hf_factory.return_value
                    hf.whoami.return_value={'name':'tester'}
                    hf.model_info.return_value.private=True
                    hf.upload_file.return_value.oid='abc123'
                    api.side_effect=[{'id':'pod1','costPerHr':0.5},None,None]
                    popen.return_value.poll.return_value=0
                    popen.return_value.returncode=0
                    launch.main()
                    self.assertEqual(api.call_args_list[0].args[:2],('POST','/pods'))
                    payload=api.call_args_list[0].args[2]
                    self.assertEqual(payload['volumeInGb'],0)
                    self.assertNotIn('terminateAfter',payload)
                    self.assertIn('POD_DEADLINE',payload['env'])
                    state=json.loads(next((root/'runs').glob('*.json')).read_text())
                    self.assertTrue(state['terminated'])
                    self.assertNotIn('test-key',json.dumps(state))
                    download.assert_called_once()
            finally:
                os.chdir(previous)

    def test_budget_includes_storage_and_reserve(self):
        for rate in [0.25,0.4,0.8,1.0]:
            for spent in [0,1,3]:
                seconds = launch.deadline_seconds(rate, spent)
                self.assertLessEqual(spent+seconds/3600*(rate+launch.STORAGE_RATE),10)
                self.assertLessEqual(seconds,8*3600)
        for rate in [0,-1,1.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): launch.deadline_seconds(rate)
        with self.assertRaises(ValueError): launch.deadline_seconds(1,9)

    def test_delete_is_verified_and_retried(self):
        with patch('launch.api', side_effect=[RuntimeError(),{},None]) as api, patch('launch.time.sleep'):
            self.assertTrue(launch.terminate('test-pod'))
            self.assertEqual(api.call_args.args,('GET','/pods/test-pod'))
        with patch('launch.api',side_effect=RuntimeError()),patch('launch.time.sleep'):
            self.assertFalse(launch.terminate('test-pod'))

    def test_watchdog_terminates_at_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json'
            path.write_text(json.dumps({'pod_id':'p1','deadline':0}))
            with patch('launch.terminate',return_value=True) as terminate:
                launch.monitor(path)
                terminate.assert_called_once_with('p1')
            self.assertTrue(json.loads(path.read_text())['terminated'])

    def test_bundle_rejects_traversal(self):
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as z: z.writestr('../escape','bad')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): safe_extract(raw.getvalue(),Path(temp))

    def test_splits_have_disjoint_tool_groups(self):
        sets=[]
        for split in ['train','validation','test']:
            rows=[json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            sets.append({tuple(sorted(t['function']['name'] for t in r['tools'])) for r in rows if r['source'].startswith('NousResearch')})
        self.assertFalse(sets[0]&sets[1]); self.assertFalse(sets[0]&sets[2]); self.assertFalse(sets[1]&sets[2])

    def test_all_prepared_calls_roundtrip_native_format(self):
        for split in ['train','validation','test','website-test']:
            for line in (ROOT/'data'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                expected=row['expected'].get('tool_calls',[])
                validate_calls(row['tools'],expected)
                actual,bad=parse_qwen(row['completion'],row['tools'])
                self.assertFalse(bad)
                self.assertEqual(normalized_calls(expected),normalized_calls(actual))

    def test_prompt_tokens_mask_cleanly(self):
        tok=Tokenizer.from_file(str(ROOT/'research/tokenizer.json'))
        rows=[json.loads(x) for x in (ROOT/'data/train.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in rows:
            p=tok.encode(row['prompt'],add_special_tokens=False).ids
            full=tok.encode(row['prompt']+row['completion'],add_special_tokens=False).ids
            self.assertEqual(full[:len(p)],p)
            self.assertGreater(len(full),len(p)); self.assertLessEqual(len(full),4096)

    def test_no_call_is_not_automatically_correct(self):
        self.assertFalse(score({'content':'Hello'},[],content='')['exact_calls'])
        self.assertFalse(score({'content':'Hello'},[],malformed=True,content='<tool_call>')['exact_calls'])

if __name__ == '__main__': unittest.main()
