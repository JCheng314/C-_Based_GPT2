#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np

B = 2
T = 64
C = 64
NH = 4
HS = C // NH
H = 4 * C
V = 50257
M = B * T


def load_tokens(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype="<u4")
    if data.size < T + 2:
        raise ValueError(f"{path} is too small for T={T}")
    if np.any(data >= V):
        raise ValueError(f"{path} contains token ids outside V={V}")
    return data.astype(np.int64)


def load_checkpoint(path: Path) -> tuple[int, dict[str, np.ndarray]]:
    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != b"CGPT2CK1":
            raise ValueError("Unsupported checkpoint magic")
        version, step, tensor_count = struct.unpack("<IiI", f.read(12))
        if version != 1:
            raise ValueError(f"Unsupported checkpoint version {version}")
        for _ in range(tensor_count):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (count,) = struct.unpack("<I", f.read(4))
            raw = f.read(count * 4)
            if len(raw) != count * 4:
                raise ValueError(f"Checkpoint ended while reading {name}")
            tensors[name] = np.frombuffer(raw, dtype="<f4").copy()
    return step, tensors


def get_batch(tokens: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, tokens.size - T - 1, size=B)
    x = np.stack([tokens[s : s + T] for s in starts]).reshape(-1)
    y = np.stack([tokens[s + 1 : s + T + 1] for s in starts]).reshape(-1)
    return x, y


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * gamma + beta


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def softmax_causal(scores: np.ndarray) -> np.ndarray:
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = scores.copy()
    scores[:, :, mask] = -1e20
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def forward_loss(tokens: np.ndarray, tensors: dict[str, np.ndarray], seed: int) -> float:
    x, y = get_batch(tokens, seed)

    token_emb = tensors["token_emb"].reshape(V, C)
    pos_emb = tensors["pos_emb"].reshape(T, C)
    hidden = token_emb[x] + np.tile(pos_emb, (B, 1))

    ln1 = layernorm(hidden, tensors["ln1_gamma"], tensors["ln1_beta"])
    qkv = ln1 @ tensors["qkv_w"].reshape(C, 3 * C) + tensors["qkv_b"]
    qkv = qkv.reshape(B, T, 3, NH, HS).transpose(0, 3, 1, 2, 4)
    q = qkv[:, :, :, 0, :]
    k = qkv[:, :, :, 1, :]
    v = qkv[:, :, :, 2, :]

    scores = np.einsum("bhtd,bhsd->bhts", q, k) / np.sqrt(float(HS))
    probs = softmax_causal(scores)
    attn_heads = np.einsum("bhts,bhsd->bhtd", probs, v)
    attn_merge = attn_heads.transpose(0, 2, 1, 3).reshape(M, C)
    h1 = attn_merge @ tensors["attn_proj_w"].reshape(C, C) + tensors["attn_proj_b"] + hidden

    ln2 = layernorm(h1, tensors["ln2_gamma"], tensors["ln2_beta"])
    mlp = gelu(ln2 @ tensors["mlp_w1"].reshape(C, H) + tensors["mlp_b1"])
    h2 = mlp @ tensors["mlp_w2"].reshape(H, C) + tensors["mlp_b2"] + h1
    logits = h2 @ tensors["lm_w"].reshape(C, V)

    logits = logits.astype(np.float64)
    row_max = logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(logits - row_max).sum(axis=1)) + row_max[:, 0]
    return float((-logits[np.arange(M), y] + logsumexp).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent NumPy baseline for train_step4 checkpoints.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    step, tensors = load_checkpoint(args.checkpoint)
    tokens = load_tokens(args.tokens)
    loss = forward_loss(tokens, tensors, args.seed)
    print(f"checkpoint_step={step} baseline_loss={loss:.6f}")


if __name__ == "__main__":
    main()
