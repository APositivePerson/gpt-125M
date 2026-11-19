"""Read-only progress view; safe to run inside the restricted sandbox."""
import json
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', type=Path, default=root / 'runs' / 'gpt2-small-100m')
args = ap.parse_args()
print(f'Run directory: {args.run_dir}')
metrics = args.run_dir / 'metrics.jsonl'
if metrics.exists():
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    for event in ['started', 'resumed', 'validation', 'resume_verified', 'train', 'stage_complete']:
        matching = [row for row in rows if row['event'] == event]
        if matching:
            print(json.dumps(matching[-1], ensure_ascii=False))
    recent = [row['tokens_per_second'] for row in rows if row['event'] == 'train'][-10:]
    train = [row for row in rows if row['event'] == 'train']
    if recent and train:
        rate = sum(recent) / len(recent)
        started = [row for row in rows if row['event'] == 'started']
        total = started[-1].get('total_tokens', 100_000_000) if started else 100_000_000
        remaining = max(0, total - train[-1]['tokens'])
        print(f'Recent mean: {rate:.0f} tokens/s; estimated remaining compute: {remaining / rate / 3600:.1f} hours (excludes data waits, evaluation and checkpoints)')
else:
    print('This run has not started yet.')
