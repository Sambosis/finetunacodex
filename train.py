"""GPU worker: BF16 LoRA, baseline/post evaluation, verified GGUF artifact backup."""
import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
if os.getenv('WANDB_API_KEY'):
    os.environ.pop('WANDB_DISABLED', None)
    os.environ.setdefault('WANDB_PROJECT', 'qwen35-web-tools')
else:
    os.environ['WANDB_DISABLED'] = 'true'
from unsloth import FastLanguageModel
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq
from huggingface_hub import HfApi
from evaluate import parse_qwen, score, summarize
from artifact_store import upload_verified
from vision_export import assert_vision_present, assert_vision_frozen, finish_vision_export

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'

def read_rows(split):
    return [json.loads(line) for line in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]

class BudgetCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Leave 90 minutes for merging, compilation, quantization, and upload.
        if time.time() >= float(os.environ['TRAIN_DEADLINE']):
            control.should_training_stop = True
        return control
    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir)/f'checkpoint-{state.global_step}'
        # Adapter backups only; no huge optimizer state uploaded every time.
        api = HfApi(token=os.environ['HF_TOKEN'])
        api.upload_folder(repo_id=os.environ['HF_REPO_ID'], folder_path=str(checkpoint),
                          path_in_repo='recovery/latest-adapter',
                          allow_patterns=['adapter*', 'tokenizer*', 'chat_template*', 'trainer_state.json', 'special_tokens*'])

def evaluate(model, tokenizer, rows, label):
    print(f"Starting {label} evaluation ({len(rows)} samples)...", flush=True)
    FastLanguageModel.for_inference(model)
    results = []
    for i, row in enumerate(rows):
        evaluation_deadline = float(os.environ['TRAIN_DEADLINE']) if label == 'baseline' else float(os.environ['POD_DEADLINE'])-3600
        if time.time() > evaluation_deadline:
            print(f"Evaluation deadline reached at sample {i}/{len(rows)}", flush=True)
            break
        inputs = tokenizer(row['prompt'], return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
        text = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
        calls, malformed = parse_qwen(text, row['tools'])
        item = score(row['expected'], calls, malformed, text)
        item.update({'id': row['source_id'], 'generated': text, 'calls': calls})
        results.append(item)
        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            print(f"[{label}] Evaluated {i+1}/{len(rows)} samples", flush=True)
    report = {'summary': summarize(results), 'results': results, 'requested_examples': len(rows), 'complete': len(results) == len(rows)}
    report['summary']['complete'] = report['complete']
    (OUT/f'{label}.json').write_text(json.dumps(report, indent=2))
    print(f"Finished {label} evaluation: {report['summary']}", flush=True)
    return report['summary']

def main():
    OUT.mkdir(exist_ok=True)
    print("Step 1/5: Loading base model and tokenizer...", flush=True)
    meta = json.loads((ROOT/'data/manifest.json').read_text())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=meta['base_model'], revision=meta['model_revision'],
        max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=False,
        load_in_16bit=True, full_finetuning=False, trust_remote_code=False,
    )
    # Keep the full image processor for saving/export; text encoding uses its tokenizer.
    if hasattr(tokenizer, 'image_processor'):
        processor = tokenizer
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(meta['base_model'], revision=meta['model_revision'], trust_remote_code=False)
    assert_vision_present(model, processor)
    template = (ROOT/'research/chat_template.jinja').read_text()
    tokenizer.chat_template = template
    processor.chat_template = template
    # Fail before training if local prepared bytes differ from the real tokenizer.
    def encode(row):
        rendered = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        if rendered != row['prompt']:
            raise ValueError('Prepared prompt differs from actual model chat template')
        prompt_ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        ids = tokenizer.encode(row['prompt'] + row['completion'], add_special_tokens=False)
        if ids[:len(prompt_ids)] != prompt_ids or len(ids) > 4096:
            raise ValueError('Token prefix mismatch or oversized sample')
        labels = [-100]*len(prompt_ids) + ids[len(prompt_ids):]
        if all(x == -100 for x in labels):
            raise ValueError('No supervised assistant tokens')
        return {'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': labels}
    print("Step 2/5: Encoding datasets...", flush=True)
    train_rows = read_rows('train')
    validation = read_rows('validation')[:128]
    train_ds = Dataset.from_list([encode(r) for r in train_rows])
    val_ds = Dataset.from_list([encode(r) for r in validation])
    test = read_rows('test')[:64] + read_rows('website-test')
    print("Step 3/5: Running baseline evaluation...", flush=True)
    before = evaluate(model, tokenizer, test, 'baseline')
    if not before['complete']:
        raise TimeoutError('Baseline did not finish within training allowance')
    upload_verified(OUT, 'artifacts')
    print("Step 4/5: Initializing LoRA adapters and Trainer...", flush=True)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        bias='none', use_gradient_checkpointing='unsloth', random_state=3407)
    assert_vision_frozen(model)
    FastLanguageModel.for_training(model)
    report_to = 'wandb' if os.getenv('WANDB_API_KEY') else 'none'
    run_name = (os.getenv('HF_REPO_ID') or 'qwen35-web-tools').split('/')[-1]
    args = TrainingArguments(output_dir=str(ROOT/'checkpoints'), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1,
        learning_rate=1e-4, lr_scheduler_type='cosine', warmup_ratio=0.05, bf16=True,
        optim='adamw_8bit', logging_steps=10, save_steps=100, save_total_limit=2,
        eval_strategy='steps', eval_steps=100, report_to=report_to, run_name=run_name, seed=3407,
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[BudgetCallback()])
    print("Starting Trainer training loop...", flush=True)
    result = trainer.train()
    (OUT/'training.json').write_text(json.dumps({'metrics': result.metrics, 'steps': trainer.state.global_step,
        'requested_epochs': 1, 'completed_epochs': trainer.state.epoch}, indent=2))
    adapter = OUT/'adapter'
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    upload_verified(adapter, 'artifacts/adapter')
    after = evaluate(model, tokenizer, test, 'finetuned')
    (OUT/'comparison.json').write_text(json.dumps({'before': before, 'after': after,
        'improved_exact_calls': after['exact_call_accuracy'] > before['exact_call_accuracy'] if after['complete'] else None,
        'note': 'Exported even if no improvement; inspect this report before adoption.'}, indent=2))
    print("Step 5/5: Exporting artifacts, merged weights, and GGUF...", flush=True)
    # Retain merged weights privately for future re-quantization.
    merged = OUT/'merged'
    model.save_pretrained_merged(str(merged), processor, save_method='merged_16bit')
    processor.save_pretrained(merged)
    upload_verified(merged, 'artifacts/merged')
    model.save_pretrained_gguf(str(OUT/'gguf'), processor, quantization_method='q4_k_m')
    files = [p for p in OUT.rglob('*[Qq]4*[Mm]*.gguf') if 'mmproj' not in p.name.lower()]
    if len(files) != 1:
        raise RuntimeError(f'Expected one language Q4_K_M GGUF; found {len(files)}')
    target = OUT/'qwen35-4b-web-tools-Q4_K_M.gguf'
    if files[0] != target:
        files[0].replace(target)
    assert_vision_frozen(model)
    finish_vision_export(OUT, meta['base_model'], meta['model_revision'])
    # Import through import_ollama.py: a language-only Modelfile loses the projector.
    (OUT/'requirements-resolved.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'], text=True))
    hashes = {}
    for path in OUT.rglob('*'):
        if path.is_file() and path.name != 'SHA256SUMS.json':
            with path.open('rb') as handle:
                hashes[path.relative_to(OUT).as_posix()] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2))
    upload_verified(OUT, 'artifacts')
    api = HfApi(token=os.environ['HF_TOKEN'])
    api.upload_file(repo_id=os.environ['HF_REPO_ID'], path_in_repo='SUCCESS.json',
        path_or_fileobj=json.dumps({'status': 'complete', 'sha256': hashes, 'finished_at': time.time()}).encode())

if __name__ == '__main__':
    main()
