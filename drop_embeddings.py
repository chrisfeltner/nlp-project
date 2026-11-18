"""Save a copy of the dataset pickle without the large `embeddings` column.

Usage (from the repository root):
    python drop_embeddings.py
    python drop_embeddings.py --src Pickles/merged_embeddings_sampled.pkl --dst Pickles/merged_sampled_no_embeddings.pkl
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

p = argparse.ArgumentParser()
p.add_argument("--src", default=str(REPO_ROOT / "Pickles" / "merged_embeddings_sampled.pkl"))
p.add_argument("--dst", default=str(REPO_ROOT / "Pickles" / "merged_sampled_no_embeddings.pkl"))
args = p.parse_args()

if Path(args.dst).resolve() == Path(args.src).resolve():
    raise SystemExit("--dst must differ from --src (the original pickle is shared, keep it)")

df = pd.read_pickle(args.src)
df.drop(columns="embeddings").to_pickle(args.dst)
print(f"{args.src} ({Path(args.src).stat().st_size / 1e9:.2f} GB) -> "
      f"{args.dst} ({Path(args.dst).stat().st_size / 1e9:.2f} GB), {len(df)} rows")
