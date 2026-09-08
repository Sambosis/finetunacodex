import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from import_ollama import verified_files
from vision_export import assert_vision_frozen, assert_vision_present, finish_vision_export

class VisionRegressionTests(unittest.TestCase):
    def test_missing_projector_refuses_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'SHA256SUMS.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'vision projector'):
                verified_files(root)

    def test_both_files_verified_and_projector_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            names=['qwen35-4b-web-tools-Q4_K_M.gguf','mmproj-F16.gguf']
            for name in names: (root/name).write_bytes(b'GGUF fixture')
            hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
            (root/'SHA256SUMS.json').write_text(json.dumps(hashes))
            self.assertEqual(set(verified_files(root)),set(names))
            (root/names[1]).write_bytes(b'GGUF corrupt')
            with self.assertRaisesRegex(RuntimeError,'SHA256'): verified_files(root)

    def test_unknown_base_cannot_borrow_projector(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'no verified fallback'):
                finish_vision_export(Path(temp),'other/model','unknown')

    def test_vision_weights_must_be_present_and_frozen(self):
        model=SimpleNamespace(config=SimpleNamespace(vision_config={'enabled':True}),
            named_parameters=lambda:[('model.visual.layer.weight',SimpleNamespace(requires_grad=False))])
        assert_vision_present(model,SimpleNamespace(image_processor=object()))
        assert_vision_frozen(model)
        model.named_parameters=lambda:[('model.visual.layer.weight',SimpleNamespace(requires_grad=True))]
        with self.assertRaisesRegex(RuntimeError,'frozen'): assert_vision_frozen(model)
        with self.assertRaisesRegex(RuntimeError,'processor'): assert_vision_present(model,object())

if __name__=='__main__': unittest.main()
