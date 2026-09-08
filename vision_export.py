"""Vision-preserving artifact packaging for text-tool LoRA training."""
import hashlib
import json
from pathlib import Path
import shutil

BASE = 'Qwen/Qwen3.5-4B'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PROJECTOR_REPO = 'unsloth/Qwen3.5-4B-GGUF'
PROJECTOR_REVISION = 'e87f176479d0855a907a41277aca2f8ee7a09523'
PROJECTOR_SHA256 = 'cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864'

def assert_vision_present(model, processor):
    if not getattr(model.config, 'vision_config', None):
        raise RuntimeError('Base model has no vision configuration')
    if not getattr(processor, 'image_processor', None):
        raise RuntimeError('Full multimodal processor is required; do not save only its tokenizer')
    names = [name for name,_ in model.named_parameters() if '.visual.' in name or '.vision_model.' in name]
    if not names:
        raise RuntimeError('Vision encoder weights are missing')

def assert_vision_frozen(model):
    changed = [name for name,p in model.named_parameters()
               if p.requires_grad and ('.visual.' in name or '.vision_model.' in name)]
    if changed:
        raise RuntimeError('Text-tool training must leave vision weights frozen: '+str(changed[:5]))

def finish_vision_export(out, base_model, base_revision):
    """Keep exporter companions; recover a frozen base projector only for exact base."""
    out = Path(out)
    projectors = [p for p in out.rglob('*.gguf') if 'mmproj' in p.name.lower()]
    if len(projectors) > 1:
        raise RuntimeError('Ambiguous vision projectors; choose the matching export explicitly')
    if projectors:
        source = projectors[0]
        target = out/'mmproj-F16.gguf'
        # Preserve its actual encoding in metadata; canonical filename is only an import name.
        if source != target:
            source.replace(target)
        origin = {'source': 'same-model Unsloth GGUF export', 'original_filename':source.name}
    else:
        if (base_model,base_revision) != (BASE,BASE_REVISION):
            raise RuntimeError('No exported vision projector, and no verified fallback for this base')
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(PROJECTOR_REPO,'mmproj-F16.gguf',revision=PROJECTOR_REVISION))
        with source.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != PROJECTOR_SHA256:
                raise RuntimeError('Pinned base projector hash mismatch')
        target = out/'mmproj-F16.gguf'
        shutil.copyfile(source,target)
        origin = {'source_repo':PROJECTOR_REPO,'source_revision':PROJECTOR_REVISION,
                  'reason':'Unmodified base vision weights; language-only LoRA. Exporter omitted companion.'}
    with target.open('rb') as stream:
        sha = hashlib.file_digest(stream,'sha256').hexdigest()
    (out/'VISION.json').write_text(json.dumps({'filename':target.name,'sha256':sha,
        'base_model':base_model,'base_revision':base_revision,**origin},indent=2))
    (out/'IMPORT.txt').write_text('Use Import-Ollama.ps1 or import_ollama.py to import BOTH GGUF files.\n'
        'A language-only FROM Modelfile will lose vision. The files API is required.\n')
    return target
