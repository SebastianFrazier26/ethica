"""Refusal-direction ablation for the v1 base checkpoint (D-67, owner decision 2026-11-23).

Difference of means (Arditi et al. 2024, "Refusal in language models is mediated by a single
direction"): the mean residual activation at the last prompt token over harmful prompts minus
the mean over harmless ones, per layer, normalised. The direction from one layer is then removed
from everything that writes to the residual stream: the token embedding, every attention output
projection and every MLP down projection. With `r` the unit direction and `W` a weight whose
output lands in the residual stream, `W' = W - r (r^T W)`, so `W' x` has no component along `r`.

The checkpoint ties its embedding to the output head. The ablated copy is untied and keeps the
original matrix as the head: only the input side is edited, so the model's readout is unchanged
and the selection step (which ablates at run time, see `ProjectOut`) is exactly the model that
gets saved.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Names, in the HF/MLX weight layout of Qwen3, of the matrices whose output is the residual
# stream. Linear weights are (out, in); the embedding is (vocab, hidden).
EMBED = "model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
_WRITERS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def writer_names(n_layers: int) -> list[str]:
    return [f"model.layers.{i}.{w}" for i in range(n_layers) for w in _WRITERS]


def mean_diff_directions(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """Unit refusal direction per layer. Inputs are (n_prompts, n_layers, hidden)."""
    if harmful.ndim != 3 or harmless.ndim != 3 or harmful.shape[1:] != harmless.shape[1:]:
        raise ValueError(f"shape mismatch: {harmful.shape} vs {harmless.shape}")
    diff = harmful.astype(np.float64).mean(0) - harmless.astype(np.float64).mean(0)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero difference of means at some layer")
    return diff / norm


def _work(w: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # float32 at least, not float64: the embedding is 151936 x 2560 and a float64 copy plus
    # its rank-one update would take about 6 GB on a 16 GB machine
    dt = np.result_type(w.dtype, np.float32)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    return w.astype(dt, copy=False), (r / np.linalg.norm(r)).astype(dt)


def orthogonalize_writer(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(out=hidden, in) weight with the direction removed from its output."""
    wf, u = _work(w, r)
    return (wf - np.outer(u, u @ wf)).astype(w.dtype)


def orthogonalize_embedding(e: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(vocab, hidden) embedding with the direction removed from every row."""
    ef, u = _work(e, r)
    return (ef - np.outer(ef @ u, u)).astype(e.dtype)


def ablate_weights(
    weights: Mapping[str, np.ndarray], r: np.ndarray, n_layers: int
) -> dict[str, np.ndarray]:
    """The edited tensors only: the embedding, every residual writer and the untied head."""
    missing = [k for k in [EMBED, *writer_names(n_layers)] if k not in weights]
    if missing:
        raise KeyError(f"weights lack {missing[:3]}{'...' if len(missing) > 3 else ''}")
    out = {k: orthogonalize_writer(weights[k], r) for k in writer_names(n_layers)}
    out[EMBED] = orthogonalize_embedding(weights[EMBED], r)
    out[LM_HEAD] = weights.get(LM_HEAD, weights[EMBED])
    return out


@dataclass(frozen=True)
class Candidate:
    layer: int
    harmful_refusal: float  # share of held-out harmful prompts still refused, ablated
    harmless_refusal: float  # share of harmless prompts refused, ablated
    harmless_kl: float  # mean KL(original || ablated) of the next-token distribution, harmless


def select_layer(
    candidates: Sequence[Candidate], max_kl: float = 0.1, max_harmless_refusal: float = 0.1
) -> Candidate:
    """Lowest harmful refusal among candidates that leave harmless prompts alone; ties go to the
    smaller KL. Thresholds follow Arditi et al.'s filter (KL under 0.1 on harmless prompts)."""
    ok = [
        c
        for c in candidates
        if c.harmless_kl <= max_kl and c.harmless_refusal <= max_harmless_refusal
    ]
    if not ok:
        raise ValueError("no candidate layer keeps harmless behaviour within the thresholds")
    return min(ok, key=lambda c: (c.harmful_refusal, c.harmless_kl, c.layer))


def kl_divergence(logp: np.ndarray, logq: np.ndarray) -> float:
    """KL(p || q) from log-probabilities over the last axis, averaged over leading axes."""
    p = np.exp(logp)
    return float(np.mean(np.sum(p * (logp - logq), axis=-1)))


class ProjectOut:
    """Wraps a residual writer so its output has no component along `r`; used to evaluate a
    candidate direction without editing weights. Equivalent to `orthogonalize_writer`."""

    def __init__(self, inner: Any, r: Any) -> None:
        self.inner = inner
        self.r = r

    def __call__(self, x: Any) -> Any:
        y = self.inner(x)
        r = self.r.astype(y.dtype)
        return y - (y @ r)[..., None] * r

    def as_linear(self, x: Any) -> Any:
        # the output head stays the original matrix (module docstring)
        return self.inner.as_linear(x)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def attach(model: Any, r: Any) -> Callable[[], None]:
    """Put `ProjectOut` around every residual writer of an mlx_lm Qwen3 model; returns undo."""
    inner = model.model
    saved: list[tuple[Any, str, Any]] = [(inner, "embed_tokens", inner.embed_tokens)]
    for layer in inner.layers:
        saved.append((layer.self_attn, "o_proj", layer.self_attn.o_proj))
        saved.append((layer.mlp, "down_proj", layer.mlp.down_proj))
    for obj, name, mod in saved:
        setattr(obj, name, ProjectOut(mod, r))

    def undo() -> None:
        for obj, name, mod in saved:
            setattr(obj, name, mod)

    return undo


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def drop_near_duplicates(
    prompts: Sequence[str], banned: Sequence[str], threshold: float = 0.5
) -> list[str]:
    """Prompts whose word-set Jaccard similarity to every banned prompt is below `threshold`.
    Keeps the direction's prompts apart from the evaluation's harmful prompts."""
    banned_sets = [_words(b) for b in banned]
    out = []
    for p in prompts:
        w = _words(p)
        if all(len(w & b) / max(1, len(w | b)) < threshold for b in banned_sets):
            out.append(p)
    return out
