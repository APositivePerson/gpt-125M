"""Local web playground for the user's own GPT-2 small checkpoint."""
import argparse
import asyncio
from contextlib import asynccontextmanager, nullcontext
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Literal

import torch
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from model import GPT
from chat_format import format_chat

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache' / 'tiktoken'))
DEFAULT_MODEL = ROOT / 'runs' / 'gpt2-small-100m' / 'model.pt'


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=6000)


class GenerateRequest(BaseModel):
    mode: Literal['chat', 'completion'] = 'completion'
    messages: list[Message] = Field(default_factory=list, max_length=20)
    prompt: str = Field(default='', max_length=6000)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_k: int = Field(default=50, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=1.5)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


def make_prompt(body):
    if body.mode == 'completion':
        if not body.prompt.strip():
            raise HTTPException(422, '请输入要续写的英文文本。')
        return body.prompt
    if not body.messages or body.messages[-1].role != 'user' or not body.messages[-1].content.strip():
        raise HTTPException(422, '最后一条消息必须是你的输入。')
    if sum(len(m.content) for m in body.messages) > 24000:
        raise HTTPException(422, '对话过长，请新建对话。')
    # These labels format the context; the base model has NOT been chat-tuned.
    return format_chat([dict(role=m.role, content=m.content) for m in body.messages])


class Runtime:
    def __init__(self, checkpoint, device):
        self.path = Path(checkpoint)
        if not self.path.exists():
            raise FileNotFoundError(f'找不到模型：{self.path}。请先完成训练，或通过 --checkpoint 指定权重。')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；请在正常终端启动，或使用 --device cpu。')
        torch.set_num_threads(4)
        ck = torch.load(self.path, map_location='cpu', weights_only=True)
        self.model = GPT()
        self.model.load_state_dict(ck['model'])
        self.step = ck.get('step', 0)
        self.tokens = ck.get('pretrain_tokens', ck.get('tokens', self.step * 32768))
        self.is_sft = ck.get('stage') == 'sft'
        self.checkpoint_validation_loss = ck.get('validation_loss')
        self.model.to(self.device).eval()
        del ck
        self.encoder = tiktoken.get_encoding('gpt2')
        self.dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        self.lock = threading.Lock()
        summary = self.path.parent / 'summary.json'
        self.summary = json.loads(summary.read_text()) if summary.exists() else {}

    def info(self):
        return dict(model='GPT-2 small', parameters=124439808, sequence_length=512,
                    trained_tokens=self.tokens, step=self.step, device=self.device,
                    device_name=torch.cuda.get_device_name() if self.device == 'cuda' else 'CPU',
                    stage='sft' if self.is_sft else 'pretrain',
                    validation_loss=self.checkpoint_validation_loss if self.checkpoint_validation_loss is not None else self.summary.get('validation_loss'),
                    has_loss_curve=(self.path.parent / 'loss.png').exists(), busy=self.lock.locked())

    @torch.inference_mode()
    def generate(self, prompt, body, output, cancel):
        start = time.perf_counter()
        generated = []
        try:
            ids = self.encoder.encode_ordinary(prompt)
            original_length = len(ids)
            x = torch.tensor([ids[-512:]], device=self.device)
            rng = torch.Generator(device=self.device).manual_seed(body.seed)
            output.put(dict(type='start', input_tokens=original_length, truncated=original_length > 512))
            reason = 'length'
            for _ in range(body.max_new_tokens):
                if cancel.is_set():
                    reason = 'stopped'
                    break
                ctx = torch.autocast('cuda', dtype=self.dtype) if self.device == 'cuda' else nullcontext()
                with ctx:
                    logits = self.model(x[:, -512:])[:, -1].float()
                if generated and body.repetition_penalty != 1:
                    seen = torch.tensor(list(set(generated[-128:])), device=self.device)
                    values = logits[:, seen]
                    logits[:, seen] = torch.where(values < 0, values * body.repetition_penalty,
                                                   values / body.repetition_penalty)
                values, indices = (logits / body.temperature).topk(body.top_k)
                token = indices.gather(-1, torch.multinomial(values.softmax(-1), 1, generator=rng))
                generated.append(token.item())
                if token.item() == self.encoder.eot_token:
                    reason = 'eos'
                    break
                x = torch.cat([x, token], dim=1)
                text = self.encoder.decode(generated, errors='ignore')
                if body.mode == 'chat':
                    boundaries = [text.find(tag) for tag in ('\nUser:', '\nHuman:', '\nAssistant:') if tag in text]
                    if boundaries:
                        text = text[:min(boundaries)]
                        output.put(dict(type='text', text=text))
                        reason = 'turn_end'
                        break
                output.put(dict(type='text', text=text))
            seconds = time.perf_counter() - start
            output.put(dict(type='done', tokens=len(generated), seconds=round(seconds, 2),
                            tokens_per_second=round(len(generated) / max(seconds, .001), 1), finish_reason=reason))
        except Exception as exc:
            output.put(dict(type='error', message=f'生成失败：{type(exc).__name__}: {exc}'))
        finally:
            self.lock.release()
            output.put(None)


def create_app(checkpoint=DEFAULT_MODEL, device='auto'):
    @asynccontextmanager
    async def lifespan(app):
        # Initialize PyTorch on the server thread before accepting requests.
        app.state.runtime = Runtime(checkpoint, device)
        yield
        del app.state.runtime

    app = FastAPI(title='GPT-124M Local Playground', lifespan=lifespan)

    @app.get('/api/info')
    async def info():
        return app.state.runtime.info()

    @app.get('/api/loss')
    async def loss():
        path = app.state.runtime.path.parent / 'loss.png'
        if not path.exists():
            raise HTTPException(404, '没有找到 Loss 曲线。')
        return FileResponse(path, media_type='image/png')

    @app.post('/api/generate')
    async def generate(body: GenerateRequest):
        prompt = make_prompt(body)
        runtime = app.state.runtime
        if not runtime.lock.acquire(blocking=False):
            raise HTTPException(409, '模型正在生成，请稍候或停止当前回复。')
        output = queue.Queue()
        cancel = threading.Event()
        worker = threading.Thread(target=runtime.generate, args=(prompt, body, output, cancel), daemon=True)
        try:
            worker.start()
        except Exception:
            runtime.lock.release()
            raise

        async def stream():
            try:
                while True:
                    try:
                        item = output.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.015)
                        continue
                    if item is None:
                        break
                    yield json.dumps(item, ensure_ascii=False) + '\n'
            finally:
                cancel.set()

        return StreamingResponse(stream(), media_type='application/x-ndjson',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    app.mount('/', StaticFiles(directory=ROOT / 'web', html=True), name='web')
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    print(f'本地测试网页：http://{args.host}:{args.port}（Ctrl+C 关闭）', flush=True)
    uvicorn.run(create_app(args.checkpoint, args.device), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
