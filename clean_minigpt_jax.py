#!/usr/bin/env python3
"""
Clean reconstruction of the code OCR'ed from DeepLearning.AI's
"Build and Train an LLM with JAX" mini-GPT lessons.

What was fixed from OCR:
- Python syntax: __init__, __call__, imports, f-strings, commas, parentheses.
- JAX syntax: jax.numpy as jnp, jnp.arange, jnp.tril, jnp.ones, jax.vmap.
- Flax NNX API spelling: nnx.Module, nnx.Embed, nnx.MultiHeadAttention,
  nnx.Linear, nnx.Rngs, nnx.Optimizer, nnx.value_and_grad, nnx.jit.
- Grain API spelling: grain.python, IndexSampler, NoSharding, Batch, DataLoader.
- Optax API spelling: optax.softmax_cross_entropy_with_integer_labels,
  optax.warmup_cosine_decay_schedule, optax.adamw.
- Orbax API spelling: orbax.checkpoint.PyTreeCheckpointer and restore args.
- Slide/narration text has been moved into comments instead of being left as code.

This file contains two usable paths:
1. A corrected Flax NNX/Grain/Optax/Orbax course-style implementation, guarded
   behind optional imports because those packages may not be installed locally.
2. A small pure-JAX fallback implementation that runs with only JAX + NumPy and
   lets this file pass a real runtime smoke test even outside the course image.

Run a smoke test:
    python clean_minigpt_jax.py --self-test

Run a tiny fallback training demo:
    python clean_minigpt_jax.py --train --file TinyStories-1000.txt

Instantiate the course-size NNX model, if optional deps are installed:
    python clean_minigpt_jax.py --show-course-model
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Lesson 1 notes: why JAX
# -----------------------------------------------------------------------------
# JAX exposes a NumPy-like API, then composes it with transformations such as
# grad, vmap, jit, and sharding. The lesson's tiny example was essentially:
# import jax.numpy as jnp
# from jax import grad
#
# def predict(params, inputs):
#     for W, b in params:
#         outputs = jnp.dot(inputs, W) + b
#         inputs = jnp.tanh(outputs)
#     return outputs
#
# def loss(params, batch):
#     inputs, targets = batch
#     preds = predict(params, inputs)
#     return jnp.sum((preds - targets) ** 2)
#
# gradient_fn = grad(loss)
# -----------------------------------------------------------------------------


def simple_predict(params: Sequence[tuple[jnp.ndarray, jnp.ndarray]], inputs: jnp.ndarray) -> jnp.ndarray:
    """Tiny JAX MLP-style example from the intro lesson."""
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
    return outputs


def simple_loss(params: Sequence[tuple[jnp.ndarray, jnp.ndarray]], batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    inputs, targets = batch
    preds = simple_predict(params, inputs)
    return jnp.sum((preds - targets) ** 2)


# -----------------------------------------------------------------------------
# Shared data/tokenization utilities
# -----------------------------------------------------------------------------
# Lesson notes:
# - TinyStories text is split by the special token <|endoftext|>.
# - The course uses the GPT-2 tokenizer through tiktoken.
# - Each story is encoded, truncated to maxlen, and padded with token 0.
# - The model is trained to predict the next token, so targets are inputs shifted
#   left by one position, with a final pad token.
# -----------------------------------------------------------------------------

SAMPLE_STORIES = [
    "Once upon a time, a small bear found a shiny rock in the forest. "
    "The bear shared it with a fox and they became friends. <|endoftext|>",
    "Lily had a red kite. The wind was strong, so her dad helped her fly it. "
    "They laughed when the kite danced above the trees. <|endoftext|>",
    "A little robot wanted to paint a flower. It mixed blue and yellow and "
    "made green leaves. Everyone clapped. <|endoftext|>",
    "The bunny ran under the apple tree. A cat jumped down, angry at the fox, "
    "and the bunny escaped. <|endoftext|>",
]


def load_stories_from_file(file_path: str | Path, max_stories: Optional[int] = None) -> list[str]:
    """Load TinyStories-style data, preserving <|endoftext|> at story boundaries.

    If the file is missing, return a small built-in sample so examples and tests run.
    """
    path = Path(file_path)
    if not path.exists():
        stories = SAMPLE_STORIES.copy()
        return stories[:max_stories] if max_stories is not None else stories

    data = path.read_text(encoding="utf-8", errors="replace")
    raw_stories = [s.strip() for s in data.split("<|endoftext|>") if s.strip()]
    stories = [s + " <|endoftext|>" for s in raw_stories]
    return stories[:max_stories] if max_stories is not None else stories


class SimpleTokenizer:
    """Small fallback tokenizer used only when tiktoken is unavailable.

    It is not GPT-2 compatible. It exists so the cleaned code can run in minimal
    environments. In the course environment, prefer tiktoken.get_encoding("gpt2").
    """

    special_tokens_set = {"<|endoftext|>"}

    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<|endoftext|>": 1, "<unk>": 2}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def n_vocab(self) -> int:
        return len(self.token_to_id)

    def _pieces(self, text: str) -> list[str]:
        return re.findall(r"<\|endoftext\|>|\w+|[^\w\s]", text, flags=re.UNICODE)

    def fit(self, texts: Iterable[str]) -> "SimpleTokenizer":
        for text in texts:
            for piece in self._pieces(text):
                if piece not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[piece] = idx
                    self.id_to_token[idx] = piece
        return self

    def encode(self, text: str, allowed_special: Optional[set[str] | list[str]] = None) -> list[int]:
        del allowed_special
        return [self.token_to_id.get(piece, self.token_to_id["<unk>"]) for piece in self._pieces(text)]

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.id_to_token.get(int(i), "<unk>") for i in ids if int(i) != 0]
        text = " ".join(pieces)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.replace(" <|endoftext|>", "")


def get_tokenizer(stories_for_fallback_fit: Optional[Sequence[str]] = None):
    """Return GPT-2 tiktoken tokenizer when available, otherwise SimpleTokenizer."""
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("gpt2")
    except Exception:
        tokenizer = SimpleTokenizer()
        if stories_for_fallback_fit:
            tokenizer.fit(stories_for_fallback_fit)
        return tokenizer


class StoryDataset:
    """Dataset that returns one fixed-length token sequence per story."""

    def __init__(self, stories: Sequence[str], maxlen: int, tokenizer) -> None:
        self.stories = list(stories)
        self.maxlen = int(maxlen)
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, idx: int) -> np.ndarray:
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})
        tokens = tokens[: self.maxlen]
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return np.asarray(tokens, dtype=np.int32)


def create_numpy_batches(
    stories: Sequence[str],
    tokenizer,
    maxlen: int,
    batch_size: int,
    shuffle: bool = False,
    num_epochs: int = 1,
    seed: int = 42,
    drop_remainder: bool = True,
) -> tuple[list[np.ndarray], int]:
    """Small replacement for Grain's IndexSampler + Batch + DataLoader."""
    dataset = StoryDataset(stories, maxlen=maxlen, tokenizer=tokenizer)
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []

    for _ in range(num_epochs):
        indices = np.arange(len(dataset))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            if drop_remainder and len(idx) < batch_size:
                continue
            batches.append(np.stack([dataset[int(i)] for i in idx], axis=0))

    batches_per_epoch = len(dataset) // batch_size if drop_remainder else math.ceil(len(dataset) / batch_size)
    return batches, batches_per_epoch


def create_grain_dataloader(
    stories: Sequence[str],
    tokenizer,
    maxlen: int,
    batch_size: int,
    shuffle: bool = False,
    num_epochs: int = 1,
    seed: int = 42,
    worker_count: int = 0,
):
    """Corrected course-style Grain data loader.

    This requires google-grain to be installed. It is intentionally not used by
    the fallback smoke test.
    """
    import grain.python as grain  # type: ignore

    dataset = StoryDataset(stories, maxlen=maxlen, tokenizer=tokenizer)
    batches_per_epoch = len(dataset) // batch_size
    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count,
    )
    return dataloader, batches_per_epoch


def load_and_preprocess_data(
    file_path: str | Path = "TinyStories-1000.txt",
    batch_size: int = 32,
    maxlen: int = 128,
    max_stories: Optional[int] = 100,
    shuffle: bool = False,
    seed: int = 42,
    prefer_grain: bool = True,
):
    """Course-style helper reconstructed from lessons 3 and 4.

    It loads TinyStories, creates a GPT-2 tokenizer when tiktoken is installed,
    and returns either a Grain DataLoader or a NumPy fallback batch list.
    """
    stories = load_stories_from_file(file_path, max_stories=max_stories)
    tokenizer = get_tokenizer(stories)

    if prefer_grain:
        try:
            return create_grain_dataloader(
                stories=stories,
                tokenizer=tokenizer,
                maxlen=maxlen,
                batch_size=batch_size,
                shuffle=shuffle,
                num_epochs=1,
                seed=seed,
                worker_count=0,
            )
        except Exception:
            pass

    return create_numpy_batches(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=maxlen,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=1,
        seed=seed,
    )


def make_target_batch(input_batch: jnp.ndarray) -> jnp.ndarray:
    """Shift every token sequence left by one position for next-token targets."""
    return jnp.concatenate(
        [input_batch[:, 1:], jnp.zeros((input_batch.shape[0], 1), dtype=input_batch.dtype)],
        axis=1,
    )


# -----------------------------------------------------------------------------
# Corrected Flax NNX course architecture
# -----------------------------------------------------------------------------
# Lesson notes:
# - MiniGPT is intentionally tiny by LLM standards: roughly 20M params with the
#   default course config below.
# - It uses learned token embeddings + learned position embeddings.
# - It uses a causal attention mask so position t cannot attend to future tokens.
# - The simplified TransformerBlock in the main lesson has multi-head attention
#   and a residual connection, but omits layer norm and the feed-forward MLP.
# -----------------------------------------------------------------------------


def causal_attention_mask(seq_len: int) -> jnp.ndarray:
    """Lower-triangular causal mask: True means attention is allowed."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


def build_nnx_classes():
    """Return corrected NNX classes. Call only when flax is installed."""
    from flax import nnx  # type: ignore

    class TokenAndPositionEmbedding(nnx.Module):
        def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs) -> None:
            self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
            self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            seq_len = x.shape[1]
            positions = jnp.arange(seq_len)[None, :]
            return self.token_emb(x) + self.pos_emb(positions)

    class TransformerBlock(nnx.Module):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            feed_forward_dim: Optional[int] = None,
            *,
            rngs,
        ) -> None:
            del feed_forward_dim  # Present in the course signature, unused in simplified block.
            self.attention = nnx.MultiHeadAttention(
                num_heads=num_heads,
                in_features=embed_dim,
                qkv_features=embed_dim,
                out_features=embed_dim,
                decode=False,
                rngs=rngs,
            )

        def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
            if mask is not None and mask.ndim == 2:
                mask = mask[None, None, :, :]
            attn_out = self.attention(x, mask=mask)
            return x + attn_out

    class MiniGPT(nnx.Module):
        def __init__(
            self,
            maxlen: int = 128,
            vocab_size: int = 50257,
            embed_dim: int = 192,
            num_heads: int = 6,
            feed_forward_dim: int = 512,
            num_transformer_blocks: int = 6,
            *,
            rngs,
        ) -> None:
            self.maxlen = maxlen
            self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
            self.transformer_blocks = [
                TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
                for _ in range(num_transformer_blocks)
            ]
            self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

        def causal_attention_mask(self, seq_len: int) -> jnp.ndarray:
            return causal_attention_mask(seq_len)

        def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
            seq_len = token_ids.shape[1]
            mask = self.causal_attention_mask(seq_len)
            x = self.embedding(token_ids)
            for block in self.transformer_blocks:
                x = block(x, mask=mask)
            logits = self.output_layer(x)
            return logits

    return TokenAndPositionEmbedding, TransformerBlock, MiniGPT


def build_course_nnx_model(
    maxlen: int = 128,
    vocab_size: int = 50257,
    embed_dim: int = 192,
    num_heads: int = 6,
    feed_forward_dim: int = 512,
    num_transformer_blocks: int = 6,
    seed: int = 0,
):
    """Instantiate the corrected Flax NNX MiniGPT model if optional deps exist."""
    from flax import nnx  # type: ignore

    _, _, MiniGPT = build_nnx_classes()
    return MiniGPT(
        maxlen=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
        rngs=nnx.Rngs(seed),
    )


def nnx_loss_fn(model, batch: tuple[jnp.ndarray, jnp.ndarray]):
    """Corrected Optax cross-entropy loss for the NNX model."""
    import optax  # type: ignore

    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets,
    ).mean()
    return loss, logits


def generate_text(
    model,
    story_prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 30,
    tokenizer=None,
    maxlen: int = 128,
    seed: int = 0,
) -> str:
    """Generic NNX/course-style autoregressive text generator.

    The model should be callable as model(input_ids) -> logits with shape
    [batch, sequence, vocab]. This mirrors the lesson-5 helper used for the
    Gradio demo.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    end_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    token_ids = tokenizer.encode(story_prompt, allowed_special={"<|endoftext|>"})[:maxlen]
    if not token_ids:
        token_ids = [0]

    key = jax.random.PRNGKey(seed)
    for _ in range(max_new_tokens):
        context = token_ids[-maxlen:]
        # Right-padding keeps position 0 aligned with the first prompt token,
        # matching the training data pipeline.
        padded = context + [0] * (maxlen - len(context))
        input_ids = jnp.asarray([padded], dtype=jnp.int32)
        logits = model(input_ids)[0, len(context) - 1]
        key, subkey = jax.random.split(key)
        if temperature <= 0:
            next_id = int(jnp.argmax(logits))
        else:
            next_id = int(jax.random.categorical(subkey, logits / temperature))
        token_ids.append(next_id)
        if next_id == end_token:
            break

    return tokenizer.decode(token_ids)


def create_nnx_optimizer_and_metrics(model, total_steps: int, warmup_ratio: float = 0.10):
    """Corrected NNX optimizer, learning-rate schedule, and loss metric."""
    from flax import nnx  # type: ignore
    import optax  # type: ignore

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=3e-4,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.01),
        wrt=nnx.Param,
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    return optimizer, metrics, lr_schedule, warmup_steps


def save_nnx_checkpoint(model, checkpoint_path: str | Path = "small_checkpoint.orbax") -> Path:
    """Corrected Orbax save snippet from lesson 4."""
    from flax import nnx  # type: ignore
    import orbax.checkpoint as ocp  # type: ignore

    path = Path(checkpoint_path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, nnx.state(model), force=True)
    return path


def restore_nnx_checkpoint_to_cpu(model, checkpoint_path: str | Path = "model_checkpoint.orbax"):
    """Corrected Orbax restore-to-CPU snippet from lesson 5."""
    from flax import nnx  # type: ignore
    import orbax.checkpoint as ocp  # type: ignore
    from jax.sharding import SingleDeviceSharding

    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=cpu_sharding),
        nnx.state(model),
    )
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        Path(checkpoint_path),
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)
    return model


# -----------------------------------------------------------------------------
# Pure-JAX fallback MiniGPT implementation
# -----------------------------------------------------------------------------
# This is not the exact course implementation, but it has the same core topology:
# token + position embeddings, causal multi-head self-attention with residuals,
# and a linear output head trained by next-token cross entropy.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class JaxMiniGPTConfig:
    maxlen: int = 32
    vocab_size: int = 128
    embed_dim: int = 32
    num_heads: int = 4
    num_transformer_blocks: int = 2

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")


def _xavier(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    fan_in = shape[0] if shape else 1
    fan_out = shape[-1] if len(shape) >= 2 else shape[0]
    scale = math.sqrt(2.0 / float(fan_in + fan_out))
    return jax.random.normal(key, shape) * scale


def init_jax_minigpt_params(config: JaxMiniGPTConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    keys = list(jax.random.split(key, 3 + 4 * config.num_transformer_blocks))
    params = {
        "token_emb": _xavier(keys.pop(), (config.vocab_size, config.embed_dim)),
        "pos_emb": _xavier(keys.pop(), (config.maxlen, config.embed_dim)),
        "blocks": [],
        "output_kernel": _xavier(keys.pop(), (config.embed_dim, config.vocab_size)),
    }
    for _ in range(config.num_transformer_blocks):
        params["blocks"].append(
            {
                "wq": _xavier(keys.pop(), (config.embed_dim, config.embed_dim)),
                "wk": _xavier(keys.pop(), (config.embed_dim, config.embed_dim)),
                "wv": _xavier(keys.pop(), (config.embed_dim, config.embed_dim)),
                "wo": _xavier(keys.pop(), (config.embed_dim, config.embed_dim)),
            }
        )
    return params


def jax_minigpt_forward(params, token_ids: jnp.ndarray, config: JaxMiniGPTConfig) -> jnp.ndarray:
    """Forward pass: token ids [B, T] -> logits [B, T, vocab]."""
    batch_size, seq_len = token_ids.shape
    del batch_size
    head_dim = config.embed_dim // config.num_heads

    x = params["token_emb"][token_ids] + params["pos_emb"][jnp.arange(seq_len)][None, :, :]
    mask = causal_attention_mask(seq_len)[None, None, :, :]

    for block in params["blocks"]:
        q = x @ block["wq"]
        k = x @ block["wk"]
        v = x @ block["wv"]

        q = q.reshape(q.shape[0], seq_len, config.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(k.shape[0], seq_len, config.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(v.shape[0], seq_len, config.num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
        scores = jnp.where(mask, scores, -1.0e9)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(x.shape[0], seq_len, config.embed_dim)
        x = x + attended @ block["wo"]

    return x @ params["output_kernel"]


def jax_cross_entropy_loss(params, input_batch: jnp.ndarray, config: JaxMiniGPTConfig) -> jnp.ndarray:
    targets = make_target_batch(input_batch)
    logits = jax_minigpt_forward(params, input_batch, config)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_losses = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    non_padding = (targets != 0).astype(jnp.float32)
    denom = jnp.maximum(non_padding.sum(), 1.0)
    return (token_losses * non_padding).sum() / denom


@partial(jax.jit, static_argnums=(3,))
def _sgd_train_step(params, input_batch: jnp.ndarray, learning_rate: float, config: JaxMiniGPTConfig):
    loss, grads = jax.value_and_grad(jax_cross_entropy_loss)(params, input_batch, config)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return updated, loss


def train_jax_fallback(
    params,
    batches: Sequence[np.ndarray],
    config: JaxMiniGPTConfig,
    steps: int = 3,
    learning_rate: float = 1e-2,
):
    """Tiny fallback trainer; enough to verify code, not enough for quality text."""
    losses: list[float] = []
    if not batches:
        raise ValueError("No batches available. Reduce batch_size or disable drop_remainder.")
    for step in range(steps):
        batch_np = batches[step % len(batches)]
        input_batch = jnp.asarray(batch_np[:, : config.maxlen], dtype=jnp.int32)
        params, loss = _sgd_train_step(params, input_batch, learning_rate, config)
        losses.append(float(loss))
    return params, losses


def generate_text_jax(
    params,
    tokenizer,
    prompt: str,
    config: JaxMiniGPTConfig,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    seed: int = 0,
) -> str:
    key = jax.random.PRNGKey(seed)
    end_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})[: config.maxlen]
    if not token_ids:
        token_ids = [0]

    for _ in range(max_new_tokens):
        context = token_ids[-config.maxlen :]
        padded = [0] * (config.maxlen - len(context)) + context
        inputs = jnp.asarray([padded], dtype=jnp.int32)
        logits = jax_minigpt_forward(params, inputs, config)[0, -1]
        key, subkey = jax.random.split(key)
        if temperature <= 0:
            next_id = int(jnp.argmax(logits))
        else:
            next_id = int(jax.random.categorical(subkey, logits / temperature))
        token_ids.append(next_id)
        if next_id == end_token:
            break

    return tokenizer.decode(token_ids)


# -----------------------------------------------------------------------------
# CLI and smoke tests
# -----------------------------------------------------------------------------


def run_self_test() -> None:
    stories = load_stories_from_file("__missing_tinystories_for_self_test__.txt", max_stories=4)
    tokenizer = get_tokenizer(stories)
    maxlen = 24
    batches, batches_per_epoch = create_numpy_batches(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=maxlen,
        batch_size=2,
        shuffle=False,
        num_epochs=1,
    )
    config = JaxMiniGPTConfig(
        maxlen=maxlen,
        vocab_size=tokenizer.n_vocab,
        embed_dim=16,
        num_heads=2,
        num_transformer_blocks=1,
    )
    params = init_jax_minigpt_params(config, seed=0)
    logits = jax_minigpt_forward(params, jnp.asarray(batches[0], dtype=jnp.int32), config)
    assert logits.shape == (2, maxlen, tokenizer.n_vocab), logits.shape
    params, losses = train_jax_fallback(params, batches, config, steps=2, learning_rate=1e-2)
    generated = generate_text_jax(
        params,
        tokenizer,
        "Once upon a time",
        config,
        max_new_tokens=8,
        temperature=0.8,
        seed=1,
    )
    print("Self-test passed.")
    print(f"Batches per epoch: {batches_per_epoch}")
    print("Losses:", ", ".join(f"{x:.4f}" for x in losses))
    print("Sample generation:", generated)


def run_train_demo(args: argparse.Namespace) -> None:
    stories = load_stories_from_file(args.file, max_stories=args.max_stories)
    tokenizer = get_tokenizer(stories)
    batches, batches_per_epoch = create_numpy_batches(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_epochs=1,
    )
    config = JaxMiniGPTConfig(
        maxlen=args.maxlen,
        vocab_size=tokenizer.n_vocab,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
    )
    params = init_jax_minigpt_params(config, seed=args.seed)
    params, losses = train_jax_fallback(
        params,
        batches,
        config,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    print(f"Loaded stories: {len(stories)}")
    print(f"Tokenizer vocab size: {tokenizer.n_vocab:,}")
    print(f"Batches per epoch: {batches_per_epoch}")
    print("Losses:", ", ".join(f"{x:.4f}" for x in losses))
    print(
        generate_text_jax(
            params,
            tokenizer,
            args.prompt,
            config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed + 1,
        )
    )


def show_course_model() -> None:
    try:
        model = build_course_nnx_model()
    except Exception as exc:
        print("Could not instantiate the course NNX model because an optional dependency is missing or incompatible.")
        print(f"Error: {type(exc).__name__}: {exc}")
        print("Install the course deps in your environment: flax, optax, orbax-checkpoint, grain, tiktoken.")
        return
    print(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleaned MiniGPT JAX course reconstruction")
    parser.add_argument("--self-test", action="store_true", help="Run a tiny pure-JAX smoke test")
    parser.add_argument("--train", action="store_true", help="Run a tiny pure-JAX training demo")
    parser.add_argument("--show-course-model", action="store_true", help="Instantiate the corrected NNX course model")
    parser.add_argument("--file", default="TinyStories-1000.txt", help="TinyStories text file")
    parser.add_argument("--max-stories", type=int, default=100)
    parser.add_argument("--maxlen", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_course_model:
        show_course_model()
    if args.self_test:
        run_self_test()
    if args.train:
        run_train_demo(args)
    if not (args.show_course_model or args.self_test or args.train):
        print("Nothing to run. Try: python clean_minigpt_jax.py --self-test")


if __name__ == "__main__":
    main()
