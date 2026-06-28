#!/usr/bin/env python3
"""
Download public TinyStories files from Hugging Face.

Examples
--------
Download the small validation split:

    python scripts/download_tinystories.py --split valid --out data/TinyStories-valid.txt

Download the full train split:

    python scripts/download_tinystories.py --split train --out data/TinyStories-train.txt

Download the GPT-4-generated V2 validation split:

    python scripts/download_tinystories.py --split v2-valid --out data/TinyStoriesV2-GPT4-valid.txt

Notes
-----
This script downloads from the public Hugging Face dataset repository:

    roneneldan/TinyStories

It does not redistribute the dataset; it only helps users fetch it themselves.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HF_REPO_ID = "roneneldan/TinyStories"
HF_REPO_TYPE = "dataset"

SPLIT_TO_FILENAME = {
    "train": "TinyStories-train.txt",
    "valid": "TinyStories-valid.txt",
    "v2-train": "TinyStoriesV2-GPT4-train.txt",
    "v2-valid": "TinyStoriesV2-GPT4-valid.txt",
    "all-tar": "TinyStories_all_data.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TinyStories files from Hugging Face."
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_TO_FILENAME),
        default="valid",
        help=(
            "Which TinyStories file to download. "
            "Use 'valid' for a small first test; 'train' is multi-GB."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. If omitted, writes to data/<upstream filename> "
            "relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help=(
            "Do not copy from the Hugging Face cache to --out. "
            "Instead, only print the cached file path."
        ),
    )
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n\n"
            "Install it with:\n\n"
            "    pip install huggingface_hub\n"
        ) from exc
    return hf_hub_download


def main() -> int:
    args = parse_args()
    filename = SPLIT_TO_FILENAME[args.split]
    out_path = args.out or Path("data") / filename

    hf_hub_download = require_huggingface_hub()

    print(f"Downloading {filename!r} from {HF_REPO_ID!r}...")
    cached_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    )

    if args.no_copy:
        print(cached_path)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing file: {out_path}\n"
            "Use --force to overwrite it."
        )

    # shutil.copyfile streams internally and avoids reading the whole file into memory.
    shutil.copyfile(cached_path, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
