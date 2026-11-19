"""Manual entry point: prepare/reuse data, then train 124M for 100M tokens."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'gpt2-small-100m')
    ap.add_argument('--log-format', choices=['progress', 'json'], default='progress')
    args = ap.parse_args()
    os.chdir(ROOT)
    subprocess.run([sys.executable, 'prepare.py', '--total-tokens', '100000000'], check=True)
    command = [sys.executable, 'train.py', '--total-tokens', '100000000',
               '--run-dir', str(args.run_dir), '--log-format', args.log_format]
    if args.resume:
        command.append('--resume')
    # Replace launcher with trainer: Ctrl+C reaches the checkpoint-save handler.
    os.execv(sys.executable, [sys.executable, *command[1:]])


if __name__ == '__main__':
    main()
