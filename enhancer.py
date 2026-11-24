import argparse
import csv
import random
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import support

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 480, 320
PATCH_SIZE = 64
PATCHES_PER_IMAGE = 32
CHANNELS = 64
WINDOW_SIZE = 8
HEADS = 4
NUM_GROUPS = 2
BLOCKS_PER_GROUP = 4
SEED = 42
LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-5
WEIGHTS = ROOT / 'best_swinir.pth'
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def partition(x, ws):
    b, h, w, c = x.shape
    return (x.reshape(b, h // ws, ws, w // ws, ws, c)
             .permute(0, 1, 3, 2, 4, 5)
             .reshape(-1, ws * ws, c))


def reverse_windows(x, ws, b, h, w):
    return (x.reshape(b, h // ws, w // ws, ws, ws, -1)
             .permute(0, 1, 3, 2, 4, 5)
             .reshape(b, h, w, -1))


class WindowAttention(nn.Module):
    def __init__(self, channels, heads, ws):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.relative_bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :] + ws - 1
        index = relative[0] * (2 * ws - 1) + relative[1]
        self.register_buffer('relative_index', index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x, mask=None):
        bw, n, c = x.shape
        qkv = self.qkv(x).reshape(bw, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relative_bias[self.relative_index.reshape(-1)]
        bias = bias.reshape(n, n, self.heads).permute(2, 0, 1)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(bw // nw, nw, self.heads, n, n)
            attn = attn + mask[None, :, None, :, :]
            attn = attn.reshape(bw, self.heads, n, n)
        out = attn.softmax(dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(bw, n, c))


class SwinBlock(nn.Module):
    def __init__(self, channels, heads, ws, shift):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, heads, ws)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 2 * channels),
                                 nn.GELU(), nn.Linear(2 * channels, channels))

    def forward(self, x, mask):
        b, h, w, c = x.shape
        z = self.norm1(x)
        if self.shift:
            z = torch.roll(z, (-self.shift, -self.shift), (1, 2))
        z = self.attn(partition(z, self.ws), mask if self.shift else None)
        z = reverse_windows(z, self.ws, b, h, w)
        if self.shift:
            z = torch.roll(z, (self.shift, self.shift), (1, 2))
        x = x + z
        return x + self.mlp(self.norm2(x))


class ResidualSwinGroup(nn.Module):
    def __init__(self, channels, heads, ws, depth):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(channels, heads, ws, 0 if i % 2 == 0 else ws // 2)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, mask):
        z = x
        for block in self.blocks:
            z = block(z, mask)
        z = self.conv(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + z


class EnhancerModel(nn.Module):
    # Lightweight SwinIR-inspired restoration network, no upsampling.
    def __init__(self, channels=CHANNELS, heads=HEADS, ws=WINDOW_SIZE,
                 groups=NUM_GROUPS, depth=BLOCKS_PER_GROUP):
        super().__init__()
        self.ws = ws
        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)
        self.groups = nn.ModuleList([
            ResidualSwinGroup(channels, heads, ws, depth) for _ in range(groups)
        ])
        self.norm = nn.LayerNorm(channels)
        self.conv_body = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.normal_(self.conv_out.weight, std=0.001)
        nn.init.zeros_(self.conv_out.bias)
        self._masks = {}

    def attention_mask(self, h, w, x):
        key = (h, w, x.device, x.dtype)
        if key not in self._masks:
            ws = self.ws
            shift = ws // 2
            regions = torch.zeros((1, h, w, 1), device=x.device, dtype=x.dtype)
            slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            for i, hs in enumerate(slices):
                for j, ts in enumerate(slices):
                    regions[:, hs, ts, :] = i * 3 + j
            ids = partition(regions, ws).squeeze(-1)
            mask = ids[:, :, None] != ids[:, None, :]
            self._masks[key] = mask.to(x.dtype) * -100.0
        return self._masks[key]

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.ws, (-w) % self.ws
        z = F.pad(x, (0, pw, 0, ph), mode='replicate') if ph or pw else x
        shallow = self.conv_in(z)
        mask = self.attention_mask(*shallow.shape[-2:], shallow)
        out = shallow.permute(0, 2, 3, 1)
        for group in self.groups:
            out = group(out, mask)
        out = self.norm(out).permute(0, 3, 1, 2)
        out = shallow + self.conv_body(out)
        return (z + self.conv_out(out))[:, :, :h, :w]

    @torch.no_grad()
    def predict(self, x, verbose=0, batch_size=1):
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = x[None]
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        if x.ndim == 3:
            x = x.unsqueeze(0)
        self.eval()
        device = next(self.parameters()).device
        out = torch.cat([self(x[i:i+1].to(device, dtype=torch.float32)).clamp(0, 1).cpu()
                         for i in range(len(x))])
        return out.numpy().transpose(0, 2, 3, 1)


def paired_paths(split):
    raw = sorted((ROOT / f'{split}rawyuv').glob('*.yuv'))
    comp = sorted((ROOT / f'{split}compyuv').glob('*.yuv'))
    if not raw or len(raw) != len(comp):
        raise ValueError(f'Missing or unmatched {split} YUV files')
    if len(raw) > 1 and [p.name for p in raw] != [p.name for p in comp]:
        raise ValueError('Multiple YUV files must have matching names')
    for a, b in zip(raw, comp):
        size = a.stat().st_size
        if size != b.stat().st_size or size % FRAME_BYTES:
            raise ValueError(f'Invalid frame count: {a.name}, {b.name}')
    return list(zip(raw, comp))


def decode_frame(buf):
    n = WIDTH * HEIGHT
    uv = n // 4
    a = np.frombuffer(buf, dtype=np.uint8)
    y = a[:n].reshape(HEIGHT, WIDTH)
    u = a[n:n+uv].reshape(HEIGHT // 2, WIDTH // 2)
    v = a[n+uv:].reshape(HEIGHT // 2, WIDTH // 2)
    return np.stack(support.yuv2rgb(y, u, v, WIDTH, HEIGHT), axis=-1)


def read_pairs(paths):
    for raw, comp in paths:
        with raw.open('rb') as r, comp.open('rb') as c:
            for _ in range(raw.stat().st_size // FRAME_BYTES):
                yield decode_frame(c.read(FRAME_BYTES)), decode_frame(r.read(FRAME_BYTES))


def load_training_data():
    paths = paired_paths('train')
    n = sum(a.stat().st_size // FRAME_BYTES for a, _ in paths)
    if n < 2:
        raise ValueError('Need at least two training images')
    val_indices = set(np.random.default_rng(SEED).choice(n, max(1, n // 10), replace=False).tolist())
    train, val = [], []
    for i, (comp, raw) in enumerate(read_pairs(paths)):
        if i in val_indices:
            val.append((comp.astype(np.float32), raw.astype(np.float32)))
        else:
            train.append((comp.astype(np.uint8), raw.astype(np.uint8)))
    print(f'Frames: train={len(train)}, validation={len(val)}', flush=True)
    return train, val


def to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255.0


class PatchDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs) * PATCHES_PER_IMAGE

    def __getitem__(self, i):
        comp, raw = self.pairs[i // PATCHES_PER_IMAGE]
        y = random.randint(0, HEIGHT - PATCH_SIZE)
        x = random.randint(0, WIDTH - PATCH_SIZE)
        comp = comp[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        raw = raw[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        # Identical spatial augmentation for both halves of the pair.
        if random.random() < 0.5:
            comp, raw = comp[:, ::-1], raw[:, ::-1]
        if random.random() < 0.5:
            comp, raw = comp[::-1], raw[::-1]
        k = random.randrange(4)
        comp, raw = np.rot90(comp, k), np.rot90(raw, k)
        return to_tensor(comp), to_tensor(raw)


def psnr_per_image(raw, out):
    return -10 * torch.log10((raw - out).square().mean(dim=(1, 2, 3)) + 1e-10)


@torch.no_grad()
def evaluate(model, pairs, device, verbose=False):
    model.eval()
    rows = []
    for i, (comp, raw) in enumerate(pairs, 1):
        comp, raw = to_tensor(comp)[None].to(device), to_tensor(raw)[None].to(device)
        out = model(comp).clamp(0, 1)
        mse = F.mse_loss(out, raw).item()
        base = psnr_per_image(raw, comp).item()
        enhanced = psnr_per_image(raw, out).item()
        rows.append((i, mse, base, enhanced, enhanced - base))
        if verbose:
            print(f'Image {i:03d} | H265 {base:.3f} | SwinIR {enhanced:.3f} | Gain {enhanced-base:+.3f} dB', flush=True)
    return np.asarray(rows)


def train_model(epochs, batch_size, device):
    set_seed()
    train, val = load_training_data()
    dataset = PatchDataset(train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED), pin_memory=device.type == 'cuda')
    model = EnhancerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE)
    best_psnr = -float('inf')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Train: {len(dataset)} random patches per epoch | batch={batch_size} | validation: {len(val)} full images', flush=True)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_mse = train_psnr = 0.0
        for step, (comp, raw) in enumerate(loader, 1):
            comp, raw = comp.to(device), raw.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(comp)
            loss = F.mse_loss(out, raw)
            loss.backward()
            optimizer.step()
            train_mse += loss.detach().item() * len(comp)
            train_psnr += psnr_per_image(raw, out.detach().clamp(0, 1)).sum().item()
            if step % 100 == 0:
                print(f'  Epoch {epoch:03d} | batch {step}/{len(loader)} | MSE {train_mse / min(step*batch_size, len(dataset)):.6f}', flush=True)
        rows = evaluate(model, val, device)
        val_mse, val_base, val_psnr, gain = rows[:, 1:].mean(axis=0)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}/{epochs} | Train MSE {train_mse/len(dataset):.6f} | Val MSE {val_mse:.6f} | '
              f'Train PSNR {train_psnr/len(dataset):.3f} | Val PSNR {val_psnr:.3f} | '
              f'Gain {gain:+.3f} dB | LR {lr:.2e} | {time.perf_counter()-started:.1f}s', flush=True)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            tmp = WEIGHTS.with_suffix('.tmp')
            torch.save(model.state_dict(), tmp)
            tmp.replace(WEIGHTS)
            print(f'  Saved {WEIGHTS.name}', flush=True)
        scheduler.step()
    print(f'Best validation PSNR: {best_psnr:.3f} dB')


def test_model(device):
    from itertools import islice

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()

    model = EnhancerModel().to(device)
    model.load_state_dict(
        torch.load(WEIGHTS, map_location=device, weights_only=True)
    )

    paths = paired_paths('test')
    rows = evaluate(
        model,
        islice(read_pairs(paths), 100),
        device,
        verbose=True
    )

    if len(rows) != 100:
        raise ValueError(f'Expected 100 test frames, found {len(rows)}')

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak_gpu_memory_gb = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
    elif device.type == 'mps':
        torch.mps.synchronize()
        peak_gpu_memory_gb = None
    else:
        peak_gpu_memory_gb = 0.0

    time_spent = time.perf_counter() - started

    with (ROOT / 'swinir_psnr.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image', 'mse', 'psnr_h265', 'psnr_swinir', 'gain'
        ])
        writer.writerows(rows.tolist())

    results = {
        'success': True,
        'psnr_enh': float(rows[:, 3].mean()),
        'time_spent': time_spent,
        'peak_gpu_memory_gb': peak_gpu_memory_gb
    }

    with (ROOT / 'results.json').open('w') as f:
        json.dump(results, f, indent=2)

    print(f'H265: {rows[:, 2].mean():.3f} dB')
    print(f'SwinIR: {rows[:, 3].mean():.3f} dB')
    print(f'Gain: {rows[:, 4].mean():+.3f} dB')
    print(f'Time: {time_spent:.2f} s')
    print('Saved results.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'], nargs='?', default='train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    parser.add_argument('--prepare', action='store_true', help='Accepted for compatibility; YUV data are loaded directly')
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else choose_device()
    print(f'Using device: {device}', flush=True)
    if args.mode == 'train':
        train_model(args.epochs, args.batch_size, device)
    else:
        test_model(device)


if __name__ == '__main__':
    main()
