"""Manual launcher: prepare English dialogues, then fine-tune your checkpoint."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--base', type=Path, default=ROOT / 'runs' / 'gpt2-small-100m' / 'model.pt')
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'gpt2-small-sft')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data_sft')
    ap.add_argument('--train-examples', type=int, default=20000)
    ap.add_argument('--val-examples', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--backend', choices=['curl', 'hub'], default='curl')
    ap.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    ap.add_argument('--local-data', type=Path)
    ap.add_argument('--proxy', help='Explicit proxy for curl dataset downloads')
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ['PYTHONUNBUFFERED'] = '1'
    if not args.resume and not args.base.exists():
        ap.error(f'Base checkpoint not found: {args.base}')
    print('[1/2] 准备对话数据；下载、读取和筛选阶段会显示进度。', flush=True)
    prepare_command = [sys.executable, '-u', str(ROOT / 'prepare_sft.py'), '--train-examples', str(args.train_examples),
                       '--val-examples', str(args.val_examples), '--output', str(args.data_dir),
                       '--backend', args.backend, '--endpoint', args.endpoint]
    if args.local_data:
        prepare_command.extend(['--local-data', str(args.local_data)])
    if args.proxy:
        prepare_command.extend(['--proxy', args.proxy])
    try:
        result = subprocess.run(prepare_command)
    except KeyboardInterrupt:
        raise SystemExit(130)
    if result.returncode:
        print('数据准备未完成，训练未启动。请按上方错误提示检查连接后重试。', flush=True)
        raise SystemExit(result.returncode if result.returncode > 0 else 130)
    print('[2/2] 启动 SFT：先加载模型并验证，然后显示训练进度条。', flush=True)
    command = [sys.executable, '-u', str(ROOT / 'train_sft.py'), '--base', str(args.base), '--run-dir', str(args.run_dir),
               '--data-dir', str(args.data_dir), '--epochs', str(args.epochs), '--lr', str(args.lr)]
    if args.resume:
        command.append('--resume')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
