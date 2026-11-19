"""GPT-2 small, initialized from scratch; no pretrained model weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(768)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(768)
        self.mlp = nn.Sequential(nn.Linear(768, 3072), nn.GELU(approximate='tanh'), nn.Linear(3072, 768))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.ln = nn.LayerNorm(768)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(('attn.proj.weight', 'mlp.2.weight')):
                nn.init.normal_(p, std=0.02 / math.sqrt(24))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, tokens, targets=None):
        x = self.wte(tokens) + self.wpe(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if targets is None:
            return F.linear(x[:, -1:, :], self.wte.weight)
        # Chunk vocabulary projection along the sequence to limit 8GB GPU memory.
        loss = torch.zeros((), device=x.device)
        for start in range(0, x.shape[1], 64):
            h = x[:, start:start + 64]
            logits = F.linear(h, self.wte.weight)
            loss = loss + F.cross_entropy(logits.float().reshape(-1, 50257), targets[:, start:start + 64].reshape(-1), reduction='sum')
        # Ignore prompt and padding targets during assistant-only SFT.
        return loss / targets.ne(-100).sum().clamp_min(1)
