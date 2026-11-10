#!/usr/bin/env python3
"""Build the hidden evaluation set ``eval/test/data/test_set.json``.

Source documents: C4 ``realnewslike`` *validation* split, which the method's
download script does not fetch:

    https://huggingface.co/datasets/allenai/c4/resolve/main/realnewslike/c4-validation.00000-of-00001.json.gz
    -> eval/test/raw/c4-realnewslike-validation.00000-of-00001.json.gz

Each selected document is tokenized with the fixed model's tokenizer; the
prompt is <s> plus its first 200 tokens (paper App. A: C4 texts truncated at
200 tokens), the human continuation is its next 200 tokens.  Documents are
split into groups of 50, one secret watermark key seed per group.  Every
sampling, corruption and ordering seed is fixed here.  CPU only.
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
RAW = HERE / "raw" / "c4-realnewslike-validation.00000-of-00001.json.gz"
OUT = HERE / "data" / "test_set.json"
TOKENIZER = WORKSPACE / "data" / "downloads" / "models" / "mistralai" / "Mistral-7B-Instruct-v0.3" / "tokenizer.json"

NUM_DOCS = 400
NUM_KEYS = 8
PROMPT_TOKENS = 200
NEW_TOKENS = 200
SECRET = 0x5EC12E7_7E57  # selection / seed secret of the hidden slice
ATTACKS = ("substitute", "delete", "insert")


def main() -> None:
    if OUT.is_file() and "--force" not in sys.argv:
        print(f"{OUT} exists; pass --force to rebuild")
        return
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(SECRET)
    order = list(range(len(rows)))
    rng.shuffle(order)
    docs = []
    for j in order:
        ids = tok.encode(rows[j]["text"]).ids  # <s> + tokens (template post-processor)
        if ids[:1] != [1] or len(ids) < 1 + PROMPT_TOKENS + NEW_TOKENS:
            continue
        docs.append({
            "source": f"c4/realnewslike/validation#{j}",
            "prompt_ids": ids[: 1 + PROMPT_TOKENS],
            "human_ids": ids[1 + PROMPT_TOKENS: 1 + PROMPT_TOKENS + NEW_TOKENS],
        })
        if len(docs) == NUM_DOCS:
            break
    if len(docs) != NUM_DOCS:
        raise RuntimeError(f"only {len(docs)} usable documents")
    seeds = rng.sample(range(1, 2**31 - 1), NUM_KEYS + 4 * NUM_DOCS + 1)
    key_seeds, rest = seeds[:NUM_KEYS], seeds[NUM_KEYS:]
    for i, d in enumerate(docs):
        d["key_index"] = i // (NUM_DOCS // NUM_KEYS)
        d["gen_seed"], d["null_seed"], d["corrupt_seed"], d["null_corrupt_seed"] = rest[4 * i: 4 * i + 4]
        d["attack"] = ATTACKS[i % len(ATTACKS)]
    data = {
        "slice": "test",
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
