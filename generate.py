"""Generate from an exported model.pt or resumable latest.pt; CPU or CUDA."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import torch
import tiktoken
from model import GPT

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=ROOT / 'runs' / 'gpt2-small-100m' / 'model.pt')
    ap.add_argument('--prompt', default='Once upon a time')
    ap.add_argument('--max-new-tokens', type=int, default=150)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--output', type=Path, help='Optional text output file')
    args = ap.parse_args()
    if args.max_new_tokens < 0 or args.temperature <= 0 or not 1 <= args.top_k <= 50257:
        ap.error('Require max-new-tokens >= 0, temperature > 0 and 1 <= top-k <= 50257')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = GPT()
    model.load_state_dict(ck['model'])
    del ck
    model.to(device).eval()
    enc = tiktoken.get_encoding('gpt2')
    ids = enc.encode_ordinary(args.prompt) or [enc.eot_token]
    x = torch.tensor([ids], device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    for _ in range(args.max_new_tokens):
        ctx = torch.autocast('cuda', dtype=dtype) if device == 'cuda' else nullcontext()
        with ctx:
            logits = model(x[:, -512:])[:, -1].float() / args.temperature
        values, indices = logits.topk(args.top_k)
        selected = torch.multinomial(values.softmax(-1), 1, generator=rng)
        token = indices.gather(-1, selected)
        x = torch.cat([x, token], dim=1)
        if token.item() == enc.eot_token:
            break
    result = enc.decode(x[0].tolist())
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')


if __name__ == '__main__':
    main()
