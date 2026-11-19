"""Prepare a small, pinned English SFT dataset. Downloads only when you run it."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import tiktoken
from chat_format import format_chat
from tqdm.auto import tqdm
from progress_status import stage
from sft_download import endpoint_url, fetch_info, curl_rows, local_files, local_identity, local_rows

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '15')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')


def encode_example(prompt, answer, encoder, seq_len=512):
    prefix = encoder.encode_ordinary(prompt)
    reply = encoder.encode_ordinary(' ' + answer.strip()) + [encoder.eot_token]
    if len(prefix) + len(reply) > seq_len + 1 or len(reply) < 2:
        return None  # Keep complete answers, including a real EOS, instead of truncating.
    ids = np.full(seq_len + 1, encoder.eot_token, dtype=np.uint16)
    mask = np.zeros(seq_len + 1, dtype=np.bool_)
    ids[:len(prefix) + len(reply)] = prefix + reply
    mask[len(prefix):len(prefix) + len(reply)] = True
    return ids, mask


def examples_from_conversation(messages):
    # System-dependent samples are omitted because this small web chat has no system role.
    if any(m.get('role') not in ('user', 'assistant') for m in messages):
        return
    history = []
    for m in messages:
        if not isinstance(m.get('content'), str) or not m['content'].strip():
            return
        expected = 'user' if len(history) % 2 == 0 else 'assistant'
        if m['role'] != expected:
            return
        if m['role'] == 'assistant':
            yield format_chat(history), m['content']
        history.append(dict(role=m['role'], content=m['content']))


def main():
    print('SFT 数据准备启动；此阶段不使用 GPU。', flush=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-examples', type=int, default=20000)
    ap.add_argument('--val-examples', type=int, default=1000)
    ap.add_argument('--seq-len', type=int, default=512, choices=[512])
    ap.add_argument('--output', type=Path, default=ROOT / 'data_sft')
    ap.add_argument('--backend', choices=['curl', 'hub'], default='curl', help='curl avoids Python requests TLS transport; SSL verification stays enabled')
    ap.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    ap.add_argument('--local-data', type=Path, help='Offline .jsonl/.parquet file, or directory of training .parquet files')
    ap.add_argument('--proxy', help='Explicit curl proxy; overrides conflicting proxy environment variables')
    args = ap.parse_args()
    try:
        args.endpoint = endpoint_url(args.endpoint)
    except ValueError as exc:
        ap.error(str(exc))
    os.environ['HF_ENDPOINT'] = args.endpoint  # Set before importing hub/datasets.
    if args.proxy:
        os.environ['SFT_CURL_PROXY'] = args.proxy
    if args.train_examples < 1 or args.val_examples < 1:
        ap.error('Example counts must be positive')
    config = dict(dataset='HuggingFaceTB/smol-smoltalk', train_examples=args.train_examples,
                  val_examples=args.val_examples, seq_len=args.seq_len, tokenizer='gpt2',
                  format='User: ...\nAssistant: ...', selection='complete replies; no system role; no truncation')
    if args.local_data:
        files = local_files(args.local_data)
        config.update(dataset='local conversations', local_source=str(args.local_data.resolve()), revision=local_identity(files))
    out = args.output
    meta = out / 'source.json'
    if meta.exists():
        previous = json.loads(meta.read_text())
        if all(previous.get(k) == v for k, v in config.items()) and all((out / f'{s}.npz').exists() for s in ('train', 'val')):
            print(f'Existing SFT data ready: {out}')
            return
        raise RuntimeError('Existing SFT data has different settings or is incomplete; use another --output directory')
    if any((out / f'{s}.npz').exists() for s in ('train', 'val')):
        raise RuntimeError('Partial SFT data found; inspect it or choose another --output directory')
    out.mkdir(parents=True, exist_ok=True)
    if args.local_data:
        rows = local_rows(files)
    elif args.backend == 'curl':
        cache = ROOT / '.cache' / 'sft_raw' / 'smol-smoltalk'
        info = fetch_info(args.endpoint, cache)
        config['revision'] = info['sha']
        rows = curl_rows(args.endpoint, cache, info)
    else:
        with stage('加载 Hugging Face 数据读取依赖'):
            from datasets import load_dataset
            from huggingface_hub import HfApi
        with stage(f'连接 Hugging Face，查询 {config["dataset"]} 版本（请求超时 30 秒）'):
            config['revision'] = HfApi(endpoint=args.endpoint).dataset_info(config['dataset'], timeout=30).sha
        with stage('打开远端数据流（需要联网读取 Parquet）'):
            rows = load_dataset(config['dataset'], split='train', revision=config['revision'], streaming=True)
    print(f'数据版本：{config["revision"]}', flush=True)
    with stage('加载 GPT-2 tokenizer'):
        encoder = tiktoken.get_encoding('gpt2')
    splits = {'train': [], 'val': []}
    seen = set()
    counts = dict(scanned=0, accepted=0, last_row=time.monotonic())
    def detail():
        return (f'已扫描 {counts["scanned"]:,} 条对话，已收集 {counts["accepted"]:,} 条样本，'
                f'距上次收到数据 {time.monotonic() - counts["last_row"]:.0f} 秒')
    # Entire conversations are assigned to one split before producing individual turns.
    with stage('读取、分词并筛选完整对话', detail=detail), tqdm(
        total=args.train_examples + args.val_examples, desc='准备 SFT 样本', unit='条', dynamic_ncols=True
    ) as bar:
        for row in rows:
            counts['scanned'] += 1
            counts['last_row'] = time.monotonic()
            if counts['scanned'] % 100 == 0:
                bar.set_postfix_str(f'扫描={counts["scanned"]:,} train={len(splits["train"]):,}/{args.train_examples:,} val={len(splits["val"]):,}/{args.val_examples:,}')
            messages = row.get('messages', [])
            digest = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            split = 'val' if int.from_bytes(digest[:4], 'big') % 20 == 0 else 'train'
            limit = args.val_examples if split == 'val' else args.train_examples
            if len(splits[split]) >= limit:
                continue
            for prompt, answer in examples_from_conversation(messages):
                encoded = encode_example(prompt, answer, encoder, args.seq_len)
                if encoded is not None and len(splits[split]) < limit:
                    splits[split].append(encoded)
                    counts['accepted'] += 1
                    bar.update(1)
            if all(len(splits[s]) >= n for s, n in [('train', args.train_examples), ('val', args.val_examples)]):
                break
    if len(splits['train']) < args.train_examples or len(splits['val']) < args.val_examples:
        raise RuntimeError(f'Insufficient fitting examples: { {s: len(v) for s, v in splits.items()} }')
    for split, samples in splits.items():
        ids, masks = zip(*samples)
        path = out / f'{split}.npz'
        with stage(f'压缩保存 {split} 数据（{len(samples):,} 条）'):
            with open(str(path) + '.tmp', 'wb') as f:
                np.savez_compressed(f, ids=np.stack(ids), mask=np.stack(masks))
            os.replace(str(path) + '.tmp', path)
        print(f'{split}: {len(samples):,} examples, {sum(int(m.sum()) for m in masks):,} assistant target tokens')
    meta.write_text(json.dumps(config, indent=2))
    print(f'SFT data saved to {out}; training has not started.')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        raise SystemExit(130)
    except Exception as exc:
        print(f'\n数据准备失败：{exc}', flush=True)
        print('尚未启动训练。检查数据源连接后重试；已完成的下载会复用。', flush=True)
        raise SystemExit(1)
