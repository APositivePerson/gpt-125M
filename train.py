import argparse
import json
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
import tiktoken
from tqdm.auto import tqdm
from model import GPT
from prepare import SHARD

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
RUN = ROOT / 'runs' / 'gpt2-small-100m'
SEQ = 512
MICRO = 4
ACCUM = 16
BATCH_TOKENS = SEQ * MICRO * ACCUM
TOTAL_STEPS = math.ceil(100_000_000 / BATCH_TOKENS)
STOP = False
LOG_JSON = False


def log(event, **fields):
    record = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), event=event, **fields)
    if LOG_JSON:
        print(json.dumps(record), flush=True)
    else:
        messages = {
            'started': lambda: f"GPT-2 124M | seq={fields['sequence_length']} | {fields['gpu']} | {fields['dtype']}",
            'resumed': lambda: f"已恢复 checkpoint：第 {fields['step']:,} 步",
            'validation': lambda: f"验证：第 {fields['step']:,} 步，loss={fields['loss']:.4f}",
            'stopped': lambda: f"已保存 checkpoint 并停止：第 {fields['step']:,} 步",
            'stage_complete': lambda: f"本阶段完成：第 {fields['step']:,} 步，验证 loss={fields['loss']:.4f}",
            'waiting_for_data': lambda: f"等待数据分片 {fields['shard']}…",
            'resume_verified': lambda: '断点恢复校验通过',
        }
        if event in messages:
            tqdm.write(messages[event](), file=sys.stderr)
    with open(RUN / 'metrics.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')


class Data:
    def __init__(self):
        self.index = None

    def batch(self, block):
        shard, pos = divmod(block, SHARD // SEQ)
        if shard != self.index:
            path = ROOT / 'data' / f'train_{shard:05d}.bin'
            waited = False
            while not path.exists():
                if STOP:
                    raise InterruptedError('Stopped while waiting for data')
                if (ROOT / 'runs' / 'producer.failed').exists():
                    raise RuntimeError('Data producer failed; see runs/prepare.log')
                if not waited:
                    log('waiting_for_data', shard=shard)
                    waited = True
                time.sleep(2)
            self.data = np.memmap(path, dtype=np.uint16, mode='r')
            self.order = np.random.default_rng(1337 + shard).permutation(SHARD // SEQ)
            self.index = shard
        at = int(self.order[pos]) * SEQ
        return np.array(self.data[at:at + SEQ + 1], dtype=np.int64)

    def get(self, first):
        z = torch.from_numpy(np.stack([self.batch(first + j) for j in range(MICRO)])).to('cuda')
        return z[:, :-1].contiguous(), z[:, 1:].contiguous()


class Engine:
    def __init__(self, total_steps=TOTAL_STEPS):
        self.total_steps = total_steps
        torch.manual_seed(1337)
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(1337)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = GPT().cuda()
        assert sum(p.numel() for p in self.model.parameters()) == 124439808
        groups = [dict(params=[p for p in self.model.parameters() if p.ndim >= 2], weight_decay=0.1),
                  dict(params=[p for p in self.model.parameters() if p.ndim < 2], weight_decay=0.)]
        self.opt = torch.optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
        self.data = Data()
        self.step = 0
        self.amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp == torch.float16)

    def save(self, name='latest.pt'):
        path = RUN / name
        torch.save(dict(model=self.model.state_dict(), optimizer=self.opt.state_dict(), scaler=self.scaler.state_dict(),
                        step=self.step, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                        config=dict(seq=SEQ, micro=MICRO, accum=ACCUM, total_steps=self.total_steps,
                                    parameters=124439808, max_positions=1024),
                        data_source=json.loads((ROOT / 'data' / 'source.json').read_text())), str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)

    def load(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['config']['seq'] == SEQ and ck['config']['micro'] == MICRO and ck['config']['accum'] == ACCUM
        if ck['config']['total_steps'] != self.total_steps:
            raise RuntimeError('Learning-rate schedule budget differs from checkpoint; use the original --total-tokens or a fresh --run-dir')
        if ck['data_source'] != json.loads((ROOT / 'data' / 'source.json').read_text()):
            raise RuntimeError('Checkpoint and local dataset identity do not match')
        self.model.load_state_dict(ck['model'])
        self.opt.load_state_dict(ck['optimizer'])
        self.scaler.load_state_dict(ck['scaler'])
        self.step = ck['step']
        torch.set_rng_state(ck['cpu_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])

    def update(self):
        self.model.train()
        warmup = min(100, max(1, self.total_steps // 10))
        if self.step < warmup:
            lr = 6e-4 * (self.step + 1) / warmup
        else:
            ratio = min(1., (self.step - warmup) / max(1, self.total_steps - warmup - 1))
            lr = 6e-5 + 0.5 * (6e-4 - 6e-5) * (1 + math.cos(math.pi * ratio))
        for group in self.opt.param_groups:
            group['lr'] = lr
        self.opt.zero_grad(set_to_none=True)
        losses = []
        for micro in range(ACCUM):
            x, y = self.data.get(self.step * MICRO * ACCUM + micro * MICRO)
            with torch.autocast('cuda', dtype=self.amp):
                loss = self.model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            losses.append(loss.detach())
            self.scaler.scale(loss / ACCUM).backward()
        self.scaler.unscale_(self.opt)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradient norm')
        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        return torch.stack(losses).mean().item(), float(norm), lr

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        val = np.memmap(ROOT / 'data' / 'val.bin', dtype=np.uint16, mode='r')
        rng = np.random.default_rng(2026)
        losses = []
        for _ in range(32):
            starts = rng.integers(0, len(val) - SEQ - 1, size=MICRO)
            z = torch.from_numpy(np.stack([np.array(val[s:s + SEQ + 1], dtype=np.int64) for s in starts])).cuda()
            with torch.autocast('cuda', dtype=self.amp):
                losses.append(self.model(z[:, :-1], z[:, 1:]).item())
        loss = sum(losses) / len(losses)
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite validation loss')
        log('validation', step=self.step, tokens=self.step * BATCH_TOKENS, loss=loss)
        return loss

    @torch.no_grad()
    def sample(self):
        self.model.eval()
        enc = tiktoken.get_encoding('gpt2')
        rng = torch.Generator(device='cuda').manual_seed(2026)
        samples = []
        for prompt in ['The solar system consists of', 'Once upon a time', 'The purpose of education is']:
            x = torch.tensor([enc.encode_ordinary(prompt)], device='cuda')
            for _ in range(80):
                with torch.autocast('cuda', dtype=self.amp):
                    logits = self.model(x[:, -SEQ:])[:, -1].float() / 0.8
                values, indices = logits.topk(50)
                selected = torch.multinomial(values.softmax(-1), 1, generator=rng)
                token = indices.gather(-1, selected)
                x = torch.cat([x, token], dim=1)
                if token.item() == enc.eot_token:
                    break
            samples.append(dict(prompt=prompt, text=enc.decode(x[0].tolist())))
        (RUN / f'samples_{self.step:06d}.json').write_text(json.dumps(samples, indent=2))
        log('samples_saved', step=self.step)

    def audit_resume(self):
        self.save('resume_audit.pt')
        a = self.update()
        expected = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.load(RUN / 'resume_audit.pt')
        b = self.update()
        max_error = max((expected[k] - v.detach().cpu()).abs().max().item() for k, v in self.model.state_dict().items())
        if abs(a[0] - b[0]) > 1e-5 or max_error > 1e-6:
            raise RuntimeError(f'Resume mismatch: {a}, {b}, max_error={max_error}')
        log('resume_verified', step=self.step, loss_error=abs(a[0] - b[0]), max_parameter_error=max_error)
        (RUN / 'resume_verified.json').write_text(json.dumps(dict(step=self.step, max_parameter_error=max_error)))
        (RUN / 'resume_audit.pt').unlink()

    def export(self):
        path = RUN / 'model.pt'
        torch.save(dict(model={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                        step=self.step, tokens=self.step * BATCH_TOKENS,
                        config=dict(parameters=124439808, layers=12, heads=12, width=768,
                                    vocab_size=50257, max_positions=1024, sequence_length=SEQ, tokenizer='gpt2'),
                        data_source=json.loads((ROOT / 'data' / 'source.json').read_text())), str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)


def main():
    global STOP, RUN, LOG_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-tokens', type=int, default=100_000_000, help='Total training token budget; rounded up to a complete optimizer batch')
    ap.add_argument('--until', type=int, help='Optional early stop step; does not change the learning-rate schedule')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--run-dir', type=Path, default=RUN, help='Checkpoint and metrics output directory')
    ap.add_argument('--log-format', choices=['progress', 'json'], default='progress', help='Terminal display (metrics.jsonl is always saved)')
    args = ap.parse_args()
    if args.total_tokens <= 0:
        ap.error('--total-tokens must be positive')
    schedule_steps = math.ceil(args.total_tokens / BATCH_TOKENS)
    args.until = schedule_steps if args.until is None else args.until
    if not 1 <= args.until <= schedule_steps:
        ap.error('--until must be between 1 and the budget step count')
    LOG_JSON = args.log_format == 'json'
    RUN = args.run_dir.resolve()
    RUN.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run outside the restricted sandbox')
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Check inputs before allocating GPU memory.
    if not args.resume and (RUN / 'latest.pt').exists():
        raise RuntimeError('Checkpoint exists; use --resume')
    if not (ROOT / 'data' / 'val.bin').exists():
        raise RuntimeError('Validation data missing; run prepare.py first')
    engine = Engine(total_steps=schedule_steps)
    if args.resume:
        engine.load(RUN / 'latest.pt')
        log('resumed', step=engine.step)
    elif (RUN / 'latest.pt').exists():
        raise RuntimeError('Checkpoint exists; use --resume')
    log('started', gpu=torch.cuda.get_device_name(), parameters=124439808, sequence_length=SEQ,
        micro_batch=MICRO, accumulation=ACCUM, batch_tokens=BATCH_TOKENS, dtype=str(engine.amp),
        total_tokens=args.total_tokens, schedule_steps=schedule_steps, until=args.until)
    initial = engine.evaluate() if engine.step == 0 else None
    if initial is not None:
        (RUN / 'initial_validation.json').write_text(json.dumps(dict(loss=initial)))
    started = time.monotonic()
    base_step = engine.step
    if args.smoke and engine.step >= 20 and not (RUN / 'resume_verified.json').exists():
        engine.audit_resume()
    with tqdm(total=args.until, initial=engine.step, desc='训练', unit='step',
              dynamic_ncols=True, disable=LOG_JSON or not sys.stderr.isatty()) as progress:
        while engine.step < args.until and not STOP:
            tic = time.monotonic()
            loss, norm, lr = engine.update()
            elapsed = time.monotonic() - tic
            progress.set_postfix_str(f'loss={loss:.3f} lr={lr:.1e} {BATCH_TOKENS / elapsed / 1000:.1f}k tok/s', refresh=False)
            progress.update(engine.step - progress.n)
            if engine.step <= 5 or engine.step % 10 == 0:
                log('train', step=engine.step, tokens=engine.step * BATCH_TOKENS, loss=loss, grad_norm=norm,
                    lr=lr, seconds=elapsed, tokens_per_second=BATCH_TOKENS / elapsed,
                    peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
            if args.smoke and (engine.step == 20 or (engine.step == 100 and not (RUN / 'resume_verified.json').exists())):
                engine.audit_resume()
                progress.update(engine.step - progress.n)
            if engine.step % 100 == 0:
                engine.save()
            if engine.step % 250 == 0:
                engine.evaluate()
                from plot_loss import plot_loss
                plot_loss(RUN)
            if engine.step % 1000 == 0:
                engine.sample()
    engine.save()
    if STOP:
        from plot_loss import plot_loss
        plot_loss(RUN)
        log('stopped', step=engine.step)
        return
    final = engine.evaluate()
    engine.sample()
    engine.export()
    from plot_loss import plot_loss
    plot_loss(RUN)
    log('stage_complete', step=engine.step, loss=final,
        average_tokens_per_second=(engine.step - base_step) * BATCH_TOKENS / (time.monotonic() - started))
    (RUN / 'summary.json').write_text(json.dumps(dict(parameters=124439808, sequence_length=SEQ,
        requested_tokens=args.total_tokens, trained_tokens=engine.step * BATCH_TOKENS,
        steps=engine.step, validation_loss=final, budget_complete=engine.step >= schedule_steps,
        model='model.pt', checkpoint='latest.pt', samples=f'samples_{engine.step:06d}.json',
        loss_curve='loss.png'), indent=2))
    if args.smoke and engine.step >= math.ceil(10_000_000 / BATCH_TOKENS):
        initial = json.loads((RUN / 'initial_validation.json').read_text())['loss']
        if final >= initial - 0.2 or not (RUN / 'resume_verified.json').exists():
            raise RuntimeError('Smoke gate failed: validation loss did not improve or resume check missing')
        (RUN / 'smoke_passed.json').write_text(json.dumps(dict(initial_loss=initial, final_loss=final,
            tokens=engine.step * BATCH_TOKENS, step=engine.step, generation='see samples JSON; early-stage quality only')))


if __name__ == '__main__':
    main()
