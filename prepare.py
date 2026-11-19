"""Stream pinned FineWeb-Edu into uint16 shards; publish only complete files."""
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
SHARD = 512 * 19532  # 10,000,384 input tokens + one next-token target
TOTAL = 100_000_000
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-tokens', type=int, default=TOTAL)
    args = ap.parse_args()
    if args.total_tokens <= 0:
        ap.error('--total-tokens must be positive')
    needed_tokens = math.ceil(args.total_tokens / 32768) * 32768
    needed_shards = math.ceil(needed_tokens / SHARD)
    OUT.mkdir(exist_ok=True)
    if (OUT / 'val.bin').exists() and (OUT / 'source.json').exists() and all(
        (OUT / f'train_{i:05d}.bin').stat().st_size == (SHARD + 1) * 2
        if (OUT / f'train_{i:05d}.bin').exists() else False for i in range(needed_shards)
    ):
        print(f'Existing data ready: {needed_shards} shards cover {needed_tokens:,} training tokens. No download needed.')
        return
    if list(OUT.glob('train_*.bin')) or (OUT / 'val.bin').exists():
        raise RuntimeError('Existing data found; refusing to overwrite. Reuse completed data or inspect partial preparation.')
    revision = HfApi().dataset_info('HuggingFaceFW/fineweb-edu').sha
    meta = dict(dataset='HuggingFaceFW/fineweb-edu', subset='sample-10BT', revision=revision,
                tokenizer='gpt2', dtype='uint16', shard_input_tokens=SHARD, target_tokens=args.total_tokens)
    (OUT / 'source.json').write_text(json.dumps(meta, indent=2))
    enc = tiktoken.get_encoding('gpt2')
    ds = load_dataset(meta['dataset'], meta['subset'], revision=revision, split='train', streaming=True)
    validation_hashes = set()
    val_tokens = 0
    buffer = []
    shard_index = 0
    start = time.monotonic()
    with open(OUT / 'val.bin.tmp', 'wb') as val:
        for row in ds:
            content = row['text']
            digest = hashlib.sha256(content.encode()).digest()
            if digest in validation_hashes:
                continue
            ids = enc.encode_ordinary(content) + [enc.eot_token]
            if val_tokens < 5_000_000:
                validation_hashes.add(digest)
                np.asarray(ids, dtype=np.uint16).tofile(val)
                val_tokens += len(ids)
                if val_tokens >= 5_000_000:
                    val.flush()
                    os.fsync(val.fileno())
                    os.replace(OUT / 'val.bin.tmp', OUT / 'val.bin')
                    print(json.dumps(dict(event='validation_ready', tokens=val_tokens)), flush=True)
                continue
            buffer.extend(ids)
            while len(buffer) >= SHARD + 1:
                path = OUT / f'train_{shard_index:05d}.bin'
                with open(str(path) + '.tmp', 'wb') as f:
                    np.asarray(buffer[:SHARD + 1], dtype=np.uint16).tofile(f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(path) + '.tmp', path)
                buffer = buffer[SHARD:]
                shard_index += 1
                print(json.dumps(dict(event='shard_ready', shards=shard_index, tokens=shard_index * SHARD,
                                      elapsed_s=round(time.monotonic() - start))), flush=True)
                if shard_index >= needed_shards:
                    (OUT / 'complete.json').write_text(json.dumps(dict(shards=shard_index, validation_tokens=val_tokens)))
                    return
    raise RuntimeError('Dataset exhausted before token budget')


if __name__ == '__main__':
    main()
