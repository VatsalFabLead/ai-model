"""Custom GPT-style decoder transformer — built from scratch in NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelConfig:
  vocab_size: int = 8192
  d_model: int = 384
  n_heads: int = 6
  n_layers: int = 6
  d_ff: int = 1536
  max_seq_len: int = 512

  @property
  def head_dim(self) -> int:
    if self.d_model % self.n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads")
    return self.d_model // self.n_heads


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  shifted = x - np.max(x, axis=axis, keepdims=True)
  exp = np.exp(shifted)
  return exp / np.sum(exp, axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
  return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return gamma * (x - mean) / np.sqrt(var + eps) + beta


class CustomTransformer:
  """Decoder-only causal transformer — fully custom, no pretrained weights."""

  def __init__(self, config: ModelConfig) -> None:
    self.config = config
    self.weights: dict[str, np.ndarray] = {}
    self._init_weights()

  def _init_weights(self) -> None:
    cfg = self.config
    scale = 0.02
    rng = np.random.default_rng(42)

    self.weights["token_emb"] = rng.normal(0, scale, (cfg.vocab_size, cfg.d_model)).astype(np.float32)
    self.weights["pos_emb"] = rng.normal(0, scale, (cfg.max_seq_len, cfg.d_model)).astype(np.float32)
    self.weights["ln_f_gamma"] = np.ones(cfg.d_model, dtype=np.float32)
    self.weights["ln_f_beta"] = np.zeros(cfg.d_model, dtype=np.float32)
    self.weights["lm_head"] = rng.normal(0, scale, (cfg.d_model, cfg.vocab_size)).astype(np.float32)

    for i in range(cfg.n_layers):
      p = f"block{i}"
      self.weights[f"{p}_ln1_gamma"] = np.ones(cfg.d_model, dtype=np.float32)
      self.weights[f"{p}_ln1_beta"] = np.zeros(cfg.d_model, dtype=np.float32)
      self.weights[f"{p}_ln2_gamma"] = np.ones(cfg.d_model, dtype=np.float32)
      self.weights[f"{p}_ln2_beta"] = np.zeros(cfg.d_model, dtype=np.float32)

      self.weights[f"{p}_attn_q"] = rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32)
      self.weights[f"{p}_attn_k"] = rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32)
      self.weights[f"{p}_attn_v"] = rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32)
      self.weights[f"{p}_attn_o"] = rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32)

      self.weights[f"{p}_ff_w1"] = rng.normal(0, scale, (cfg.d_model, cfg.d_ff)).astype(np.float32)
      self.weights[f"{p}_ff_b1"] = np.zeros(cfg.d_ff, dtype=np.float32)
      self.weights[f"{p}_ff_w2"] = rng.normal(0, scale, (cfg.d_ff, cfg.d_model)).astype(np.float32)
      self.weights[f"{p}_ff_b2"] = np.zeros(cfg.d_model, dtype=np.float32)

  def _attention(self, x: np.ndarray, block_idx: int, causal_mask: np.ndarray) -> np.ndarray:
    cfg = self.config
    p = f"block{block_idx}"
    q = x @ self.weights[f"{p}_attn_q"]
    k = x @ self.weights[f"{p}_attn_k"]
    v = x @ self.weights[f"{p}_attn_v"]

    batch, seq, _ = q.shape
    hd = cfg.head_dim
    nh = cfg.n_heads

    q = q.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, nh, hd).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
    scores = scores + causal_mask[:seq, :seq]
    attn = _softmax(scores, axis=-1)
    out = attn @ v
    out = out.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.d_model)
    return out @ self.weights[f"{p}_attn_o"]

  def _ffn(self, x: np.ndarray, block_idx: int) -> np.ndarray:
    p = f"block{block_idx}"
    h = x @ self.weights[f"{p}_ff_w1"] + self.weights[f"{p}_ff_b1"]
    h = _gelu(h)
    return h @ self.weights[f"{p}_ff_w2"] + self.weights[f"{p}_ff_b2"]

  def forward(self, token_ids: np.ndarray) -> np.ndarray:
    """Return logits shape (batch, seq, vocab_size)."""
    cfg = self.config
    batch, seq = token_ids.shape
    if seq > cfg.max_seq_len:
      raise ValueError(f"Sequence length {seq} exceeds max_seq_len {cfg.max_seq_len}")

    x = self.weights["token_emb"][token_ids] + self.weights["pos_emb"][:seq]
    x = x[np.newaxis, ...] if x.ndim == 2 else x

    causal = np.triu(np.full((cfg.max_seq_len, cfg.max_seq_len), -1e9, dtype=np.float32), k=1)

    for i in range(cfg.n_layers):
      p = f"block{i}"
      ln1 = _layer_norm(x, self.weights[f"{p}_ln1_gamma"], self.weights[f"{p}_ln1_beta"])
      x = x + self._attention(ln1, i, causal)
      ln2 = _layer_norm(x, self.weights[f"{p}_ln2_gamma"], self.weights[f"{p}_ln2_beta"])
      x = x + self._ffn(ln2, i)

    x = _layer_norm(x, self.weights["ln_f_gamma"], self.weights["ln_f_beta"])
    return x @ self.weights["lm_head"]

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = np.array(
      [
        self.config.vocab_size,
        self.config.d_model,
        self.config.n_heads,
        self.config.n_layers,
        self.config.d_ff,
        self.config.max_seq_len,
      ],
      dtype=np.int32,
    )
    np.savez_compressed(path, meta=meta, **self.weights)

  @classmethod
  def load(cls, path: Path) -> CustomTransformer:
    data = np.load(path, allow_pickle=False)
    meta = data["meta"]
    config = ModelConfig(
      vocab_size=int(meta[0]),
      d_model=int(meta[1]),
      n_heads=int(meta[2]),
      n_layers=int(meta[3]),
      d_ff=int(meta[4]),
      max_seq_len=int(meta[5]),
    )
    model = cls(config)
    for key in model.weights:
      model.weights[key] = data[key]
    return model

  def get_weights_dict(self) -> dict[str, np.ndarray]:
    return self.weights

  def set_weights_dict(self, weights: dict[str, np.ndarray]) -> None:
    self.weights = weights
