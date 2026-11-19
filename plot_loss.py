"""Render actual training/validation metrics, usable while training is running."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(run_dir, title='GPT-2 small (124M) / FineWeb-Edu'):
    run_dir = Path(run_dir)
    metrics = run_dir / 'metrics.jsonl'
    if not metrics.exists():
        return
    records = {}
    for line in metrics.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A concurrently written last line may be incomplete.
        if row.get('event') in ('train', 'validation'):
            records[(row['event'], row['step'])] = row
    train = sorted((r for (e, _), r in records.items() if e == 'train'), key=lambda r: r['step'])
    val = sorted((r for (e, _), r in records.items() if e == 'validation'), key=lambda r: r['step'])
    if not train and not val:
        return
    with open(run_dir / 'loss.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'tokens', 'train_loss', 'validation_loss', 'learning_rate'])
        for step in sorted({s for _, s in records}):
            t = records.get(('train', step), {})
            v = records.get(('validation', step), {})
            writer.writerow([step, t.get('tokens', v.get('tokens', '')), t.get('loss', ''), v.get('loss', ''), t.get('lr', '')])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    if train:
        ax.plot([r['tokens'] / 1e6 for r in train], [r['loss'] for r in train],
                color='#2563eb', linewidth=1.1, alpha=.8, label='Training loss (logged batches)')
    if val:
        ax.plot([r['tokens'] / 1e6 for r in val], [r['loss'] for r in val],
                'o-', color='#ea580c', markersize=4, linewidth=1.6, label='Held-out validation loss')
    ax.set(xlabel='Processed tokens (millions)', ylabel='Cross-entropy loss (nats)', title=title)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(run_dir / 'loss.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'gpt2-small-100m')
    args = ap.parse_args()
    plot_loss(args.run_dir)
    print(f'Loss curve: {args.run_dir / "loss.png"}')
