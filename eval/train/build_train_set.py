#!/usr/bin/env python3
"""Build the public development set ``eval/train/data/train_set.json``.

Source documents: the first usable documents of the public C4 ``realnewslike``
train shard that ``scripts/download.sh`` fetches:

    data/downloads/c4_realnewslike/c4-train.00000-of-00512.json.gz

Same construction as the held-out slice (prompt = <s> + first 200 tokens,
human continuation = next 200 tokens, groups of 50 documents per key seed),
with 200 documents and 4 public key seeds.  CPU only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RAW = WORKSPACE / "data" / "downloads" / "c4_realnewslike" / "c4-train.00000-of-00512.json.gz"
OUT = HERE / "data" / "train_set.json"
TOKENIZER = WORKSPACE / "data" / "downloads" / "models" / "mistralai" / "Mistral-7B-Instruct-v0.3" / "tokenizer.json"

NUM_DOCS = 200
NUM_KEYS = 4
PROMPT_TOKENS = 200
NEW_TOKENS = 200
PUBLIC_SEED = 20250123
ATTACKS = ("substitute", "delete", "insert")


def main() -> None:
    if OUT.is_file() and "--force" not in sys.argv:
        print(f"{OUT} exists; pass --force to rebuild")
        return
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    docs = []
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        for j, line in enumerate(f):
            ids = tok.encode(json.loads(line)["text"]).ids
            if ids[:1] != [1] or len(ids) < 1 + PROMPT_TOKENS + NEW_TOKENS:
                continue
            docs.append({
                "source": f"c4/realnewslike/train-00000#{j}",
                "prompt_ids": ids[: 1 + PROMPT_TOKENS],
                "human_ids": ids[1 + PROMPT_TOKENS: 1 + PROMPT_TOKENS + NEW_TOKENS],
            })
            if len(docs) == NUM_DOCS:
                break
    if len(docs) != NUM_DOCS:
        raise RuntimeError(f"only {len(docs)} usable documents")
    rng = random.Random(PUBLIC_SEED)
    seeds = rng.sample(range(1, 2**31 - 1), NUM_KEYS + 4 * NUM_DOCS + 1)
    key_seeds, rest = seeds[:NUM_KEYS], seeds[NUM_KEYS:]
    for i, d in enumerate(docs):
        d["key_index"] = i // (NUM_DOCS // NUM_KEYS)
        d["gen_seed"], d["null_seed"], d["corrupt_seed"], d["null_corrupt_seed"] = rest[4 * i: 4 * i + 4]
        d["attack"] = ATTACKS[i % len(ATTACKS)]
    data = {
        "slice": "train",
        "prompt_tokens": PROMPT_TOKENS,
        "max_new_tokens": NEW_TOKENS,
        "key_seeds": key_seeds,
        "detect_order_seed": rest[-1],
        "docs": docs,
    }
    data["digest"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"wrote {OUT}: {len(docs)} docs, {NUM_KEYS} keys")


if __name__ == "__main__":
    main()
