# GPT-2 small / FineWeb-Edu / 100M tokens

Train a **124,439,808 parameter** GPT-2 small from random weights: 12 layers,
12 heads, width 768, GPT-2 tokenizer (50,257 vocabulary), tied embeddings.
Sequence length is **512**; the model reserves 1,024 positional embeddings.

## Optional next stage: English conversation SFT (you launch it)

To teach the existing checkpoint the dialogue format, stop the web server first to
free GPU memory, then run:

```bash
cd /home/wangziyi/GPT-124M
.venv/bin/python launch_sft.py
```

This downloads and prepares **20,000 training examples + 1,000 validation examples**
from [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
then fine-tunes **your existing** `runs/gpt2-small-100m/model.pt` for **2 epochs**.
No other pretrained model weights are used. It starts only when you run the command.
The default microbatch is 4, accumulation 4, sequence length 512 and peak LR 5e-5.
These are initial experimental settings, not a guarantee of dialogue quality.

The launcher labels data preparation and training separately. Preparation prints
network/tokenizer stages and a sample-count progress bar showing scanned, training
and validation counts. Operations without a known percentage print a heartbeat
every 10 seconds. Hugging Face metadata requests have a 30-second request timeout.
Training shows model-loading stages, a validation progress bar, save status and
then optimizer-step progress. During initial data preparation GPU use can be zero.
After updating these scripts, interrupt the old process and rerun to see new status
messages; already running Python code does not reload edits automatically.

The dataset is an English SFT collection adapted for smaller models. Preparation
pins its revision, omits system-dependent conversations, holds out whole conversations
by content hash, and keeps only examples whose complete reply fits 512 tokens.
The code never fabricates an end-of-answer token at an arbitrary truncation point.
The template `User: ...\nAssistant: ...` is shared with the web server. Only assistant
reply tokens and their true EOT token contribute to loss; user prompts and padding
are masked. The base model and its original outputs are preserved.

Outputs go to `runs/gpt2-small-sft/`: `latest.pt` for resume, `best_model.pt` selected
by validation assistant loss, `model.pt` for the final weights, `loss.png`, `loss.csv`,
`metrics.jsonl`, `sft_config.json`, and `summary.json`. Validation uses a fixed subset
of up to 256 held-out examples. Its **assistant-only loss is not directly comparable**
to the original full-text pretraining loss.

To resume, use the same settings:

```bash
.venv/bin/python launch_sft.py --resume
```

To test the best checkpoint in the web page afterward:

```bash
.venv/bin/python web_chat.py --checkpoint runs/gpt2-small-sft/best_model.pt
```

The page detects SFT metadata and switches to chat mode. Output can be up to 2,048
new tokens, but the trained context remains 512. SFT targets dialogue formatting and
response style; it cannot compensate for limited pretraining knowledge or provide
reliable Chinese ability. Compare held-out prompts and base/SFT responses before
deciding whether the fine-tune improved your intended use.

Separate entries are `prepare_sft.py` and `train_sft.py`; each supports `--help`.

### If SFT data download fails with SSL / unexpected EOF or HTTP 401

The default download transport is now **curl**, with normal TLS certificate checks,
three retries, visible file progress, and `.part` resume files. It queries the public
repository version, pins that revision, then downloads training Parquet shards into
`.cache/sft_raw/smol-smoltalk/`. Subsequent filtering reads local files. Already
downloaded shards and pinned metadata are reused.

For shard requests the downloader asks for a fresh signed redirect each attempt.
Dataset files must use `/datasets/HuggingFaceTB/smol-smoltalk/resolve/<revision>/...`;
omitting `/datasets/` addresses a model repository and can produce a misleading 401.
HTTP 401/403 refreshes that redirect up to three times; it does not disable TLS or
assume that a public dataset requires a token. A successful homepage request alone
does not verify the redirected file-storage host. Unsupported range resumes preserve
the partial file under a `.restart-*` name and retry from the beginning.

Run the same command to retry:

```bash
.venv/bin/python launch_sft.py
```

To choose your local proxy explicitly instead of relying on proxy environment
variables, add `--proxy http://127.0.0.1:7897`.

`SSL: UNEXPECTED_EOF_WHILE_READING` occurs before training; it means the TLS
connection was interrupted. Turning off certificate checks is not part of this fix.
If curl also fails, verify your connection/proxy or explicitly choose an accessible
HTTPS Hub-compatible endpoint with `--endpoint https://your-endpoint.example`.
The `HF_ENDPOINT` environment variable is also honored. A mirror that redirects
back to the official host may not resolve the connectivity issue.

Alternatively, supply downloaded **training** data locally, with no Hugging Face
metadata or data request:

```bash
.venv/bin/python launch_sft.py --local-data /path/to/train.parquet
```

`--local-data` accepts one Parquet/JSONL file, or a directory of training Parquet
files. Each row must contain `messages`, e.g.
`{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello!"}]}`.
The GPT-2 tokenizer must already be cached (it is on this machine).
`--backend hub` retains the original Python Hugging Face streaming transport as an
optional alternative. Data preparation failures stop the launcher without starting
training or printing a second redundant Python traceback.

## Start training yourself

Stop any older training process first with **Ctrl+C**, and wait for its checkpoint
confirmation. Then, from a normal terminal with access to your GPU:

```bash
cd /home/wangziyi/GPT-124M
.venv/bin/python launch.py
```

This prepares or reuses data and runs training in the foreground, with a progress
bar showing loss, learning rate, speed and estimated remaining time. The existing
dataset already covers this budget, so no new download is needed on this machine.
The new output directory is `runs/gpt2-small-100m`; old experiments are preserved.
Nothing starts automatically just because the scripts have been edited.

The run targets **100M tokens**, rounded up to **100,007,936 tokens / 3,052 steps**
to complete the last optimizer batch. At the observed ~15k tokens/s, allow roughly
two hours including checkpoint, evaluation and plotting overhead. This is a short
pretraining experiment; generated text may still be repetitive or inconsistent.

To resume after Ctrl+C or another interruption:

```bash
.venv/bin/python launch.py --resume
```

Direct training entry point, if data is already prepared:

```bash
.venv/bin/python train.py --total-tokens 100000000 --run-dir runs/gpt2-small-100m
```

Add `--resume` and keep the same token budget and output directory when resuming.
The learning-rate schedule budget is saved in the checkpoint and checked on load.
Older 2.5B-budget checkpoints use a different schedule; this 100M experiment starts
in its own directory from random weights.

## Outputs

Training automatically writes these files in `runs/gpt2-small-100m/`:

| File | Purpose |
| --- | --- |
| `latest.pt` | Resumable model, optimizer, scaler, step, RNG and data identity; saved every 100 steps, at the end and after Ctrl+C |
| `model.pt` | Final inference weights, architecture metadata and data identity, without optimizer state |
| `samples_003052.json` | Final generated samples for three prompts |
| `loss.png` | Training and held-out validation cross-entropy curves |
| `loss.csv` | Numeric values used for the curves |
| `metrics.jsonl` | Detailed training metrics, including throughput and GPU memory |
| `summary.json` | Final step count, token count, validation loss and artifact names |

Validation and Loss plots update every 250 steps; generation samples are also saved
every 1,000 steps. At completion the code saves the checkpoint, evaluates, generates
text, exports inference weights, draws the final curve, and writes the summary.
Stopping early saves a checkpoint and the available Loss curve; final inference
weights are exported when the requested stage completes.

## Generate text separately

### Local web playground

```bash
cd /home/wangziyi/GPT-124M
.venv/bin/python web_chat.py
```

Open **http://127.0.0.1:8000** in your browser. The server automatically uses CUDA
when available, loads `runs/gpt2-small-100m/model.pt` once, and runs inference on
your own machine. Ctrl+C in the terminal shuts down the web server.

The page includes chat-style testing, direct text continuation, streaming output,
stop/new conversation buttons, temperature, Top-K, repetition penalty, generation
length and seed settings, inference timing, model statistics and a Loss-curve link.
The checkpoint is an **English base model**, without instruction or chat tuning.
Chat mode formats messages as `User: ... / Assistant: ...`; this does not add chat
training. For a representative test, use **text continuation** with prompts such as
`The purpose of education is` or `Once upon a time`.

The page defaults to text continuation and a **512-token generation budget**, adjustable
up to **2,048 new tokens**. This is an upper bound: EOT or a chat role boundary may end
generation sooner, and the UI shows the reason. The trained context window remains
**512 tokens**; longer output uses a sliding window, which can lose earlier context.
Increasing the output limit does not add knowledge, Chinese training, or chat ability.

Context is limited to the most recent 512 tokens, including the conversation labels.
The server allows one generation at a time; stopping a reply cancels generation
after the currently running token step. Chat history stays in page memory and is
cleared on refresh. No external model service is used.

Optional startup settings:

```bash
.venv/bin/python web_chat.py --device cpu --port 8001
.venv/bin/python web_chat.py --checkpoint runs/gpt2-small-100m/latest.pt
```

### Command-line generation

After training finishes:

```bash
.venv/bin/python generate.py --prompt "Once upon a time" --max-new-tokens 150
```

To generate during training, read the latest atomic checkpoint:

```bash
.venv/bin/python generate.py --checkpoint runs/gpt2-small-100m/latest.pt --device cpu --prompt "The solar system"
```

Use `--output generated.txt` to save the result. Generation loads only the model
weights; it supports CPU or CUDA, temperature, top-k and a fixed sampling seed.

To redraw the Loss curve at any time:

```bash
.venv/bin/python plot_loss.py --run-dir runs/gpt2-small-100m
```

## Data and optimization

`HuggingFaceFW/fineweb-edu`, `sample-10BT`, pinned revision in `data/source.json`.
The first whole documents totaling at least 5M tokens form validation; their content
hashes are excluded from training. Subsequent documents are tokenized with GPT-2,
separated by EOT, and written into atomic uint16 shards. Blocks are deterministically
shuffled inside each shard. Existing prepared data is reused; larger source data
does not increase the training token budget.

Microbatch 4, accumulation 16: **32,768 tokens per optimizer step**. BF16 autocast,
FP32 weights and AdamW states, SDPA causal attention, gradient clipping 1.0.
AdamW uses betas (0.9, 0.95), weight decay 0.1, peak LR 6e-4. Warmup lasts 100
steps, followed by cosine decay to 6e-5 over this **100M-token budget**.
Validation repeatedly measures the same deterministic 65,536-token held-out sample.
No pretrained model weights are downloaded.

Dependencies are in `requirements.txt`; `.venv` reuses the existing system PyTorch.
Dataset, tokenizer and Matplotlib caches live under `.cache/` in this project.
# gpt-125M
