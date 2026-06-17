"""Full backpropagation trainer for the custom transformer (NumPy only).

Implements real gradients (attention, layernorm, GELU FFN, embeddings) and an
Adam optimizer. Saved weights are 100% compatible with `CustomTransformer.load`.
No third-party models or pretrained weights are used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.engine.tokenizer import ByteTokenizer
from app.engine.transformer import CustomTransformer, ModelConfig


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  shifted = x - np.max(x, axis=axis, keepdims=True)
  exp = np.exp(shifted)
  return exp / np.sum(exp, axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
  c = math.sqrt(2.0 / math.pi)
  return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
  c = math.sqrt(2.0 / math.pi)
  inner = c * (x + 0.044715 * x**3)
  t = np.tanh(inner)
  dinner = c * (1.0 + 3.0 * 0.044715 * x**2)
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner


def _layernorm_fwd(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  inv_std = 1.0 / np.sqrt(var + eps)
  xhat = (x - mean) * inv_std
  y = gamma * xhat + beta
  cache = (xhat, gamma, inv_std)
  return y, cache


def _layernorm_bwd(dy: np.ndarray, cache):
  xhat, gamma, inv_std = cache
  d = xhat.shape[-1]
  dgamma = np.sum(dy * xhat, axis=0)
  dbeta = np.sum(dy, axis=0)
  dxhat = dy * gamma
  dx = (inv_std / d) * (
    d * dxhat
    - np.sum(dxhat, axis=-1, keepdims=True)
    - xhat * np.sum(dxhat * xhat, axis=-1, keepdims=True)
  )
  return dx, dgamma, dbeta


class Trainer:
  """Trains a CustomTransformer with full backprop and Adam."""

  def __init__(self, model: CustomTransformer) -> None:
    self.model = model
    self.cfg = model.config
    self.W = model.weights
    self._m = {k: np.zeros_like(v) for k, v in self.W.items()}
    self._v = {k: np.zeros_like(v) for k, v in self.W.items()}
    self._t = 0

  def _causal_mask(self, seq: int) -> np.ndarray:
    return np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)

  def _attn_fwd(self, a: np.ndarray, p: str, mask: np.ndarray):
    cfg = self.cfg
    nh, hd, seq = cfg.n_heads, cfg.head_dim, a.shape[0]
    q = a @ self.W[f"{p}_attn_q"]
    k = a @ self.W[f"{p}_attn_k"]
    v = a @ self.W[f"{p}_attn_v"]

    qh = q.reshape(seq, nh, hd).transpose(1, 0, 2)
    kh = k.reshape(seq, nh, hd).transpose(1, 0, 2)
    vh = v.reshape(seq, nh, hd).transpose(1, 0, 2)

    scale = 1.0 / math.sqrt(hd)
    scores = (qh @ kh.transpose(0, 2, 1)) * scale + mask
    attn = _softmax(scores, axis=-1)
    ctx = attn @ vh
    ctx_concat = ctx.transpose(1, 0, 2).reshape(seq, cfg.d_model)
    out = ctx_concat @ self.W[f"{p}_attn_o"]

    cache = (a, qh, kh, vh, attn, ctx_concat, scale)
    return out, cache

  def _attn_bwd(self, dout: np.ndarray, cache, p: str):
    cfg = self.cfg
    nh, hd = cfg.n_heads, cfg.head_dim
    a, qh, kh, vh, attn, ctx_concat, scale = cache
    seq = a.shape[0]
    g: dict[str, np.ndarray] = {}

    g[f"{p}_attn_o"] = ctx_concat.T @ dout
    dctx_concat = dout @ self.W[f"{p}_attn_o"].T
    dctx = dctx_concat.reshape(seq, nh, hd).transpose(1, 0, 2)

    dattn = dctx @ vh.transpose(0, 2, 1)
    dvh = attn.transpose(0, 2, 1) @ dctx

    dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))

    dqh = (dscores @ kh) * scale
    dkh = (dscores.transpose(0, 2, 1) @ qh) * scale

    dq = dqh.transpose(1, 0, 2).reshape(seq, cfg.d_model)
    dk = dkh.transpose(1, 0, 2).reshape(seq, cfg.d_model)
    dv = dvh.transpose(1, 0, 2).reshape(seq, cfg.d_model)

    g[f"{p}_attn_q"] = a.T @ dq
    g[f"{p}_attn_k"] = a.T @ dk
    g[f"{p}_attn_v"] = a.T @ dv

    da = dq @ self.W[f"{p}_attn_q"].T + dk @ self.W[f"{p}_attn_k"].T + dv @ self.W[f"{p}_attn_v"].T
    return da, g

  def _ffn_fwd(self, x: np.ndarray, p: str):
    h_pre = x @ self.W[f"{p}_ff_w1"] + self.W[f"{p}_ff_b1"]
    h = _gelu(h_pre)
    out = h @ self.W[f"{p}_ff_w2"] + self.W[f"{p}_ff_b2"]
    return out, (x, h_pre, h)

  def _ffn_bwd(self, dout: np.ndarray, cache, p: str):
    x, h_pre, h = cache
    g: dict[str, np.ndarray] = {}
    g[f"{p}_ff_w2"] = h.T @ dout
    g[f"{p}_ff_b2"] = np.sum(dout, axis=0)
    dh = dout @ self.W[f"{p}_ff_w2"].T
    dh_pre = dh * _gelu_grad(h_pre)
    g[f"{p}_ff_w1"] = x.T @ dh_pre
    g[f"{p}_ff_b1"] = np.sum(dh_pre, axis=0)
    dx = dh_pre @ self.W[f"{p}_ff_w1"].T
    return dx, g

  def forward(self, ids: np.ndarray):
    cfg = self.cfg
    seq = len(ids)
    x = self.W["token_emb"][ids] + self.W["pos_emb"][:seq]
    mask = self._causal_mask(seq)

    block_caches = []
    for i in range(cfg.n_layers):
      p = f"block{i}"
      ln1, ln1_c = _layernorm_fwd(x, self.W[f"{p}_ln1_gamma"], self.W[f"{p}_ln1_beta"])
      attn_out, attn_c = self._attn_fwd(ln1, p, mask)
      x2 = x + attn_out
      ln2, ln2_c = _layernorm_fwd(x2, self.W[f"{p}_ln2_gamma"], self.W[f"{p}_ln2_beta"])
      ff_out, ff_c = self._ffn_fwd(ln2, p)
      x3 = x2 + ff_out
      block_caches.append((ln1_c, attn_c, ln2_c, ff_c))
      x = x3

    xf, lnf_c = _layernorm_fwd(x, self.W["ln_f_gamma"], self.W["ln_f_beta"])
    logits = xf @ self.W["lm_head"]
    cache = (ids, xf, lnf_c, block_caches)
    return logits, cache

  def backward(self, logits: np.ndarray, cache):
    cfg = self.cfg
    ids, xf, lnf_c, block_caches = cache
    seq = len(ids)
    grads: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.W.items()}

    probs = _softmax(logits, axis=-1)
    dlogits = probs.copy()
    n_targets = seq - 1
    loss = 0.0
    for t in range(n_targets):
      target = ids[t + 1]
      loss -= math.log(probs[t, target] + 1e-9)
      dlogits[t, target] -= 1.0
    dlogits[seq - 1] = 0.0
    dlogits /= max(n_targets, 1)
    loss /= max(n_targets, 1)

    grads["lm_head"] = xf.T @ dlogits
    dxf = dlogits @ self.W["lm_head"].T
    dx, dgf, dbf = _layernorm_bwd(dxf, lnf_c)
    grads["ln_f_gamma"] = dgf
    grads["ln_f_beta"] = dbf

    for i in reversed(range(cfg.n_layers)):
      p = f"block{i}"
      ln1_c, attn_c, ln2_c, ff_c = block_caches[i]
      dff_out = dx
      dln2, gff = self._ffn_bwd(dff_out, ff_c, p)
      grads.update(gff)
      dx2_a, dg2, db2 = _layernorm_bwd(dln2, ln2_c)
      grads[f"{p}_ln2_gamma"] = dg2
      grads[f"{p}_ln2_beta"] = db2
      dx2 = dx + dx2_a

      dattn_out = dx2
      dln1, gattn = self._attn_bwd(dattn_out, attn_c, p)
      grads.update(gattn)
      dx1_a, dg1, db1 = _layernorm_bwd(dln1, ln1_c)
      grads[f"{p}_ln1_gamma"] = dg1
      grads[f"{p}_ln1_beta"] = db1
      dx = dx2 + dx1_a

    np.add.at(grads["token_emb"], ids, dx)
    grads["pos_emb"][:seq] += dx
    return loss, grads

  def step(self, grads: dict[str, np.ndarray], lr: float, clip: float = 1.0,
           beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    total = math.sqrt(sum(float(np.sum(g**2)) for g in grads.values()))
    scale = clip / (total + 1e-6) if total > clip else 1.0

    self._t += 1
    bc1 = 1.0 - beta1**self._t
    bc2 = 1.0 - beta2**self._t
    for k, g in grads.items():
      g = g * scale
      self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
      self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g * g)
      m_hat = self._m[k] / bc1
      v_hat = self._v[k] / bc2
      self.W[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def build_sequences(text: str, tokenizer: ByteTokenizer, seq_len: int) -> list[np.ndarray]:
  ids = tokenizer.encode(text, add_bos=True, add_eos=True)
  sequences: list[np.ndarray] = []
  step = max(1, seq_len // 2)
  for start in range(0, max(1, len(ids) - 1), step):
    chunk = ids[start : start + seq_len + 1]
    if len(chunk) >= 2:
      sequences.append(np.array(chunk, dtype=np.int64))
  return sequences


def train_model(
  corpus_path: Path,
  weights_out: Path,
  tokenizer_out: Path,
  config: ModelConfig,
  epochs: int = 30,
  learning_rate: float = 3e-3,
  seq_len: int = 48,
) -> CustomTransformer:
  """Train custom model on local corpus with real backprop — no external models."""
  text = corpus_path.read_text(encoding="utf-8")
  tokenizer = ByteTokenizer.train_from_text(text, vocab_size=config.vocab_size)
  tokenizer.save(tokenizer_out)

  model = CustomTransformer(config)
  trainer = Trainer(model)
  sequences = build_sequences(text, tokenizer, seq_len)
  if not sequences:
    raise ValueError("Corpus too small for training. Add more text to data/corpus.txt")

  rng = np.random.default_rng(0)
  for epoch in range(epochs):
    order = rng.permutation(len(sequences))
    total_loss = 0.0
    for idx in order:
      ids = sequences[idx]
      logits, cache = trainer.forward(ids)
      loss, grads = trainer.backward(logits, cache)
      trainer.step(grads, lr=learning_rate)
      total_loss += loss
    avg = total_loss / len(sequences)
    if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
      print(f"Epoch {epoch + 1}/{epochs} — loss: {avg:.4f}")

  model.save(weights_out)
  return model
