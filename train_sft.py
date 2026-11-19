"""Assistant-only supervised fine-tuning of YOUR 124M checkpoint; manual launch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from model import GPT
from plot_loss import plot_loss
from progress_status import stage

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def atomic_save(payload, path):
    with stage(f'保存 {path.name}'):
        torch.save(payload, str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)


def batch(data, indices, device):
    ids = torch.from_numpy(data['ids'][indices].astype(np.int64)).to(device)
    mask = torch.from_numpy(data['mask'][indices, 1:]).to(device)
    x = ids[:, :-1].contiguous()
    y = ids[:, 1:].contiguous().masked_fill(~mask, -100)
    return x, y


def lr_at(step, total, peak):
    warmup = min(50, max(1, total // 20))
    if step < warmup:
        return peak * (step + 1) / warmup
    ratio = min(1., (step - warmup) / max(1, total - warmup - 1))
    return peak * (.1 + .45 * (1 + math.cos(math.pi * ratio)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=ROOT / 'runs' / 'gpt2-small-100m' / 'model.pt')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data_sft')
    ap.add_argument('--run-dir', type=Path, default=ROOT / 'runs' / 'gpt2-small-sft')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--micro-batch', type=int, default=4)
    ap.add_argument('--accumulation', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    print('SFT 训练进程已启动，正在检查 GPU 与训练数据。', flush=True)
    if min(args.epochs, args.micro_batch, args.accumulation) < 1 or args.lr <= 0:
        ap.error('Epochs, batch sizes and learning rate must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. Start training in your normal GPU-enabled terminal.')
    out = args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'latest.pt').exists() and not args.resume:
        raise RuntimeError('SFT checkpoint exists; use --resume or a new --run-dir')
    for name in ('train.npz', 'val.npz', 'source.json'):
        if not (args.data_dir / name).exists():
            raise RuntimeError('SFT data missing. Run prepare_sft.py first.')
    data = {}
    for split in ('train', 'val'):
        with stage(f'读取 {split} 数据'):
            with np.load(args.data_dir / f'{split}.npz', allow_pickle=False) as f:
                data[split] = dict(ids=f['ids'], mask=f['mask'])
        d = data[split]
        if d['ids'].shape != d['mask'].shape or d['ids'].shape[1] != 513 or not np.all(d['mask'][:, 1:].any(axis=1)):
            raise RuntimeError(f'Invalid {split} tensors or empty assistant target')
    n = len(data['train']['ids'])
    effective = args.micro_batch * args.accumulation
    steps_per_epoch = math.ceil(n / effective)
    total = steps_per_epoch * args.epochs
    identity = json.loads((args.data_dir / 'source.json').read_text())
    data_hash = hashlib.sha256()
    for split in ('train', 'val'):
        data_hash.update(data[split]['ids'].tobytes())
        data_hash.update(data[split]['mask'].tobytes())
    config = dict(epochs=args.epochs, micro_batch=args.micro_batch, accumulation=args.accumulation,
                  lr=args.lr, total_steps=total, data=identity, data_sha256=data_hash.hexdigest(), seq=512)
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    print(f'GPU：{torch.cuda.get_device_name()}；训练 {n:,} 条 × {args.epochs} epochs = {total:,} 步', flush=True)
    with stage('初始化 124M 模型并分配显存'):
        model = GPT().cuda()
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=.01),
        dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.)
    ], lr=args.lr, betas=(.9, .95), fused=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    state = dict(step=0, tokens=0, best_loss=float('inf'))
    if args.resume:
        with stage('读取续训 checkpoint'):
            ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        if ck['sft_config'] != config:
            raise RuntimeError('SFT configuration or data changed; resume with the original arguments and files')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        state.update(ck['state'])
        pretrain_tokens = ck['pretrain_tokens']
        torch.set_rng_state(ck['cpu_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        del ck
    else:
        with stage(f'加载基础模型权重：{args.base.name}'):
            ck = torch.load(args.base, map_location='cpu', weights_only=True)
            model.load_state_dict(ck['model'])
        pretrain_tokens = ck.get('pretrain_tokens', ck.get('tokens', ck.get('step', 0) * 32768))
        del ck
    (out / 'sft_config.json').write_text(json.dumps(config, indent=2))
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def log(event, **fields):
        with open(out / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(dict(event=event, **fields)) + '\n')
    def weights(validation_loss=None):
        return dict(model={k: p.detach().cpu() for k, p in model.state_dict().items()},
                    stage='sft', step=state['step'], tokens=state['tokens'], pretrain_tokens=pretrain_tokens,
                    config=dict(parameters=124439808, sequence_length=512, tokenizer='gpt2', chat_template='User: ...\nAssistant: ...'),
                    data_source=identity, validation_loss=validation_loss)
    def save():
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         stage='sft', step=state['step'], tokens=state['tokens'], pretrain_tokens=pretrain_tokens,
                         state=state.copy(), sft_config=config, cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all()), out / 'latest.pt')
    @torch.no_grad()
    def evaluate():
        model.eval()
        numerator = 0.; denominator = 0
        # Fixed held-out subset, weighted by actual assistant target tokens.
        validation_size = min(256, len(data['val']['ids']))
        for start in tqdm(range(0, validation_size, args.micro_batch), desc='SFT 验证', unit='batch', leave=False, dynamic_ncols=True):
            indices = np.arange(start, min(start + args.micro_batch, 256, len(data['val']['ids'])))
            x, y = batch(data['val'], indices, 'cuda')
            with torch.autocast('cuda', dtype=dtype):
                loss = model(x, y)
            count = int(y.ne(-100).sum())
            numerator += float(loss) * count
            denominator += count
        result = numerator / denominator
        if not math.isfinite(result):
            raise RuntimeError('Non-finite SFT validation loss')
        log('validation', step=state['step'], tokens=state['tokens'], loss=result)
        tqdm.write(f'SFT 验证：step={state["step"]:,}, assistant loss={result:.4f}')
        if result < state['best_loss']:
            state['best_loss'] = result
            atomic_save(weights(result), out / 'best_model.pt')
        plot_loss(out, title='GPT-2 small (124M) / assistant-only SFT')
        model.train()
        return result
    log('started', total_steps=total, examples=n, stage='sft')
    if state['step'] == 0:
        print('先做基础验证，完成后开始参数更新。', flush=True)
        evaluate()
    try:
        with tqdm(total=total, initial=state['step'], desc='对话微调', unit='step', dynamic_ncols=True) as bar:
            while state['step'] < total and not stopping:
                epoch, within_epoch = divmod(state['step'], steps_per_epoch)
                order = np.random.default_rng(1337 + epoch).permutation(n)
                indices = order[within_epoch * effective:(within_epoch + 1) * effective]
                chunks = [indices[s:s + args.micro_batch] for s in range(0, len(indices), args.micro_batch)]
                target_count = sum(int(data['train']['mask'][chunk, 1:].sum()) for chunk in chunks)
                optimizer.zero_grad(set_to_none=True)
                lr = lr_at(state['step'], total, args.lr)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                total_loss = 0.
                tic = time.perf_counter()
                for chunk in chunks:
                    x, y = batch(data['train'], chunk, 'cuda')
                    count = int(y.ne(-100).sum())
                    with torch.autocast('cuda', dtype=dtype):
                        loss = model(x, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite SFT training loss')
                    scaler.scale(loss * count / target_count).backward()
                    total_loss += float(loss.detach()) * count / target_count
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite SFT gradients')
                scaler.step(optimizer)
                scaler.update()
                state['step'] += 1
                state['tokens'] += len(indices) * 512
                log('train', step=state['step'], tokens=state['tokens'], loss=total_loss, lr=lr, grad_norm=float(norm))
                bar.set_postfix_str(f'loss={total_loss:.3f} lr={lr:.1e} {len(indices) * 512 / (time.perf_counter() - tic):.0f} tok/s', refresh=False)
                bar.update(1)
                if state['step'] % 200 == 0:
                    evaluate()
                    save()
                elif state['step'] % 100 == 0:
                    save()
        final_loss = evaluate()
        save()
        atomic_save(weights(final_loss), out / 'model.pt')
        (out / 'summary.json').write_text(json.dumps(dict(stage='sft', validation_loss=final_loss,
            best_validation_loss=state['best_loss'], trained_tokens=state['tokens'], steps=state['step'],
            completed=state['step'] == total, assistant_only_loss=True, pretrain_tokens=pretrain_tokens), indent=2))
        print(f'{"已保存并停止" if stopping else "微调完成"}：{out}\n网页测试：.venv/bin/python web_chat.py --checkpoint {out / "best_model.pt"}')
    except BaseException:
        # A normal Ctrl+C sets the stop flag; preserve the prior atomic checkpoint on unexpected errors.
        raise


if __name__ == '__main__':
    main()
