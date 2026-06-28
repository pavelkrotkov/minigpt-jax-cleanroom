#!/usr/bin/env python3
"""
Create a small TinyStories subset file while preserving story boundaries.

Examples
--------
Create the course-style 1000-story file from the validation split:

    python scripts/make_tinystories_1000.py \
      --input data/TinyStories-valid.txt \
      --output data/TinyStories-1000.txt \
      --n 1000

Create a 100-story smoke-test file:

    python scripts/make_tinystories_1000.py \
      --input data/TinyStories-valid.txt \
      --output data/TinyStories-100.txt \
      --n 100

Notes
-----
TinyStories stores stories separated by the GPT-2 end-of-text token:

    <|endoftext|>

This script streams the input file, so it can operate on the full train file
without loading the full multi-GB dataset into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator


END_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller TinyStories text file with N complete stories."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input TinyStories .txt file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/TinyStories-1000.txt"),
        help="Output subset path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of complete stories to write.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help=(
            "Strip leading/trailing whitespace from each story before writing. "
            "By default, internal formatting from the source file is preserved."
        ),
    )
    return parser.parse_args()


def iter_stories(path: Path, *, strip: bool = False) -> Iterator[str]:
    """Yield complete TinyStories records including the END_TOKEN.

    The implementation is streaming and handles the case where a line contains
    zero, one, or multiple END_TOKEN occurrences.
    """
    current: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(END_TOKEN)

            # No complete story ended on this line.
            if len(parts) == 1:
                current.append(line)
                continue

            # Every part before the last is followed by END_TOKEN.
            for part in parts[:-1]:
                current.append(part)
                story = "".join(current)
                if strip:
                    story = story.strip()

                if story:
                    yield story + END_TOKEN

                current = []

            # The final part is the beginning of the next story, if nonempty.
            if parts[-1]:
                current.append(parts[-1])

    # TinyStories normally ends with END_TOKEN. If it does not, preserve the
    # final partial record as a story and add END_TOKEN for consistency.
    if current:
        story = "".join(current)
        if strip:
            story = story.strip()
        if story:
            yield story + END_TOKEN


def main() -> int:
    args = parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be a positive integer.")

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing file: {args.output}\n"
            "Use --force to overwrite it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8", newline="") as out:
        for story in iter_stories(args.input, strip=args.strip):
            out.write(story)
            if not story.endswith("\n"):
                out.write("\n")
            count += 1

            if count >= args.n:
                break

    if count < args.n:
        print(
            f"Warning: requested {args.n:,} stories, but only found {count:,} "
            f"in {args.input}."
        )

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {count:,} stories to {args.output} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
