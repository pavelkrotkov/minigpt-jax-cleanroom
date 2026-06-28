# MiniGPT JAX Cleanroom

Unofficial cleaned single-file MiniGPT/JAX companion inspired by DeepLearning.AI's
"Build and Train an LLM with JAX" short course.

This repo provides a runnable Python script for:

- token and position embeddings
- causal self-attention
- MiniGPT-style next-token modeling
- TinyStories loading
- small CPU training
- optional Flax NNX / Grain / Optax / Orbax course-style path
- minimal pure-JAX fallback path

This is not an official course repo. It does not include course videos, transcripts,
screenshots, proprietary notebooks, or pretrained checkpoints.

## Quick start

```bash
git clone ...
cd minigpt-jax-cleanroom
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python clean_minigpt_jax.py --self-test
```

## Get TinyStories

This repo does not commit TinyStories data files. Download them locally from the public Hugging Face dataset and create a small course-style subset.

Install the downloader dependency:

```bash
pip install huggingface_hub
````

Download the smaller validation split first:

```bash
python scripts/download_tinystories.py \
  --split valid \
  --out data/TinyStories-valid.txt
```

Create a 1,000-story subset:

```bash
python scripts/make_tinystories_1000.py \
  --input data/TinyStories-valid.txt \
  --output data/TinyStories-1000.txt \
  --n 1000
```

Then train the tiny demo:

```bash
python clean_minigpt_jax.py \
  --train \
  --file data/TinyStories-1000.txt
```

For the full training file:

```bash
python scripts/download_tinystories.py \
  --split train \
  --out data/TinyStories-train.txt
```

The full train file is large. Start with `valid` unless you specifically want the full dataset.

