# Implementation Notes

## Why this is a single Python script

The course is notebook-based. This repo consolidates the flow into a single script
so it can be run from the command line, tested, and version-controlled cleanly.

## Why there is a pure-JAX fallback

The course stack depends on Flax NNX, Grain, Optax, Orbax, and tiktoken.
The fallback path lets users verify the architecture and training loop with only JAX and NumPy.

## Padding convention

Training uses right-padding. Generation also right-pads and reads logits from the last real token.

## Temperature sampling

The course-style notebook used argmax generation. This repo uses categorical sampling when
temperature is positive, so temperature actually changes generation behavior.
