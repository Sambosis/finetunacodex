# Qwen3.5-4B tool training for Local Chat

This is a script-based Unsloth BF16 LoRA training job for the website at
`W:\home\sambo\projects\web-starter`. The website itself was not edited.

The saved training run `20260907-163552-8ab4bc` produced a tool adapter and merged
checkpoint. Its original Ollama export omitted the vision projector. On September
8, the local export was repaired without retraining: the adapter was confirmed to
contain only language-layer changes, and the pinned base projector was restored.
The repaired model passed three real image probes and a structured tool-call probe.
The revised future GPU export path has local regression checks but has not itself
been exercised by a new paid training run.

## Vision preservation

`Import-Ollama.ps1` now uses `import_ollama.py` to supply BOTH language and projector
GGUFs to Ollama's files API, and verifies `vision` and `tools` capabilities. Do not
use the old language-only Modelfile to import this model. The importer refuses
missing or checksum-mismatched projectors. The local repair's `VISION.json` records
the projector's pinned source and hash; the original language GGUF is unchanged.

Training keeps the multimodal processor, checks that vision weights exist, and
checks that no vision weights are trainable. Exports preserve their own projector;
only the exact pinned Qwen3.5-4B base may use the verified frozen-base fallback.
This preserves image input support; it does not prove unchanged accuracy across
all vision tasks after language fine-tuning. Image regression probes are in
`test_vision.py`, and `test_vision_pipeline.py` covers missing/tampered artifacts.

## Start

Add these values to `.env` in this folder (never paste keys into chat):

```dotenv
RUNPOD_API_KEY=your_key
HF_TOKEN=your_hugging_face_write_token
```

The Hugging Face token needs permission to create and write a private model repo
under your account. The RunPod key needs permission to create, inspect, and delete
pods. Secrets are not saved in the run ledger or training bundle. RunPod receives
them as pod environment variables; the trainer does not receive the RunPod key.

From PowerShell in this directory:

```powershell
.\Run-Training.ps1 -Preflight   # Free local validation; no rental
.\Run-Training.ps1              # Rent, train, export, back up, delete, download
```

No browser or notebook is involved. Keep the launcher running for automatic local
download. If your computer disconnects, the remote supervisor continues the job
and attempts to delete its own pod; retrieve the artifacts later with:

```powershell
.\.venv\Scripts\python.exe launch.py --download YOUR_NAME/REPO_FROM_RUN_LEDGER
```

After a successful run, import and run the website-contract evaluation:

```powershell
.\Import-Ollama.ps1 -ArtifactDirectory '.\artifacts\RUN_ID\artifacts'
```

Select `qwen35-web-tools:4b` in the website's existing model picker. The import
uses `RENDERER qwen3.5` and `PARSER qwen3.5`, verified in your installed Ollama
0.33.3 base model. The website uses `think: false` and real structured
`message.tool_calls`, which is also how the evaluation script makes requests.
The importer evaluates but does not automatically declare the new model better
or change the website's chosen model. Evaluation calls never execute the tools.

## Data and training

Source: [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1),
Apache 2.0, revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`.
Nous identifies it as the structured-output/function-calling data used for Hermes
2 Pro. This pipeline uses the multi-turn and cleaned Glaive subsets. It does not
use the overlapping single-turn file or copy the Hermes chat serialization.

Preparation validates function names, arguments, JSON schemas, and tool-result
pairing; rejects malformed examples; removes exact conversation duplicates; and
groups identical tool-name sets into one split. It converts structured calls
through the pinned **Qwen3.5 XML function/parameter template**. Overlength samples
are excluded instead of cutting a call in half. User prompts and tool results
are masked from training loss; each completion trains only one assistant turn.

Prepared pilot:

| Split | Assistant turns | Purpose |
|---|---:|---|
| Training | 4,212 | 4,000 public turns plus 212 synthetic website turns |
| Validation | 952 | Loss evaluation pool; 128 used during the pilot |
| Public test | 882 | 64 used for before/after generation evaluation |
| Website test | 23 | Held-out tool choice, arguments, clarification, no-call, and tool-result probes |

Website schemas were extracted from the actual calculator, time, notes, Python /
JavaScript runner, search, and webpage-reader definitions. Synthetic examples are
a small adaptation curriculum, not evidence of benchmark performance. The
split rule reduces direct leakage but does not eliminate semantic overlap or
pretraining contamination. Exact-call scoring penalizes alternative equivalent
arguments, and no-call scoring does not grade the factual quality of an answer.

Recipe: pinned `Qwen/Qwen3.5-4B`, BF16, rank 16 LoRA, alpha 32, 4,096-token
training length, batch 1 with accumulation 8, learning rate 1e-4, one epoch.
Train loss, adapter checkpoints, baseline/post reports, merged BF16 weights,
Q4_K_M GGUF, Modelfile, and SHA256 hashes are retained in a **private** HF repo.
The fixed website supports longer contexts, but this experiment only trains at
4,096 tokens. It targets text tool use while retaining the frozen vision encoder.

## Spending and cleanup

The total authorization is **$14**. The launcher reserves **$4** as headroom and
will allocate at most $10 across ledger-recorded attempts. It queries Secure
Cloud A40, RTX A6000, and RTX 4090 prices, accepts only a quoted GPU rate at or
below $1/hour, and checks the actual price returned on creation. There is a
maximum eight-hour lifetime, including startup. Only one pod is requested; no
spot retries, persistent volumes, or blind retries of an ambiguous create call.
100 GB of temporary container storage is included in the conservative estimate.
Previous attempts consume their full reserved estimate even if they ended early.

Training stops early enough to leave 90 minutes for evaluation/export/upload.
A remote watchdog starts before package installation; a separate local watchdog
also deletes the pod at its absolute deadline. Completion/failure cleanup checks
that deletion actually succeeded. Upload verification checks both sizes and
hashes before the success marker is written. At the deadline, protecting the
budget takes priority over finishing the model; uploaded recovery adapters may
be all that is available if training or export fails.

**These are application-level controls, not a guaranteed RunPod billing cap.**
RunPod's documented timer flags were [removed because they were not enforced](https://github.com/runpod/runpodctl/pull/330).
Simultaneous host/network/control-plane failures can prevent deletion. There is
no honest way for this script to guarantee a strict $14 bill under every outage.
Unrelated workloads and account auto-recharge are outside this script's control.

If a run has an unknown creation/deletion outcome, the ledger blocks another
launch. Reconcile the exact named pod in the RunPod console before changing the
ledger. Never clear the ledger just to bypass the budget checks.

## Reproduce local checks

```powershell
node export-tools.cjs 'W:\home\sambo\projects\web-starter'
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe make_website_tests.py
.\.venv\Scripts\python.exe -m unittest test_pipeline -v
```

`research/requirements-gpu-resolved.txt` was resolved with uv for Linux x86_64 /
Python 3.12 without renting a GPU. The full lock avoids incompatible newer
Transformers/datasets versions that exceed Unsloth's published constraints.
At runtime on the GPU pod, `supervisor.py` uses `uv` for parallel dependency
installation and `hf_transfer` for base model weight downloads, cutting
pre-training startup from ~25 minutes down to ~2 minutes.
The Docker image tag was checked for existence. These checks do not replace a
CUDA smoke test. The baseline and fine-tuned GPU evaluations run in the paid job;
The repaired local model has additionally passed the image and tool smoke probes
recorded in `artifacts/vision-verification/results.json`.

References: [Unsloth Qwen3.5 training](https://unsloth.ai/docs/models/qwen3.5/fine-tune),
[RunPod lifecycle API](https://docs.runpod.io/api-reference/pods/POST/pods),
[Ollama model import](https://docs.ollama.com/import).
