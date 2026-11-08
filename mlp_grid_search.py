"""Grid search over MLP hyperparameters for the three encoders used in Experiments.ipynb.

Uses the notebook's data setup (Left/Right labels, seeded 20% sample, 90/10 train/test split) and searches on a
smaller sample of the train split, divided into train/validation. The test set is never used.
Each setting keeps the epoch with the lowest validation loss (as part1.smlp does); settings are ranked by
validation macro F1. Results are written to grid_search/results.csv after every run.

Run from the repo root: python mlp_grid_search.py
"""
import contextlib
import io
import itertools
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from part1 import smlp

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "grid_search"                      # result CSVs, tracked in git
MODEL_DIR = ROOT / "results" / "grid_search"    # scratch model file, overwritten every run (gitignored)
LABEL_COL = "Party_orientation"
SAMPLE_FRAC = 0.2       # same as the notebook
GRID_SAMPLE = 30_000    # rows of the notebook's train split used for the search
SEED = 0

GRID = {
    "hidden_dim": [64, 128, 256],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "batch_size": [128, 512],
}
FIXED = {"weight_decay": 1e-4, "num_epochs": 25}

orientation_to_side = {
    "Centre-left to left": "Left",
    "Centre-left": "Left",
    "Centre to centre-left": "Left",
    "Centre to centre-right": "Right",
    "Centre-right": "Right",
    "Right": "Right",
    "Right to far-right": "Right",
    "Far-right": "Right",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# encoders, as in Experiments.ipynb
def sparse_row(counts, size):
    indices = sorted(counts)
    values = np.array([counts[i] for i in indices], dtype=np.float32)
    return csr_matrix((values, indices, [0, len(indices)]), shape=(1, size))

class TokenIndicator:
    def __init__(self, vocab):
        self.vocab = vocab
    def __len__(self):
        return len(self.vocab)
    def __call__(self, x):
        indices = {self.vocab[item] for item in x.split() if item in self.vocab}
        return sparse_row({index: 1 for index in indices}, len(self))

class BagOfWords(TokenIndicator):
    def __call__(self, x):
        counts = Counter(self.vocab[item] for item in x.split() if item in self.vocab)
        return sparse_row(counts, len(self))

class Embedding:
    dense = True
    def __init__(self, frame):
        self.lookup = {s: np.asarray(e, dtype=np.float32) for s, e in zip(frame["sentence"], frame["embeddings"])}
    def __len__(self):
        return 1024
    def __call__(self, x):
        return self.lookup[x]


def get_vocab(texts, min_count=5, max_size=None):
    counts = Counter(item for text in texts for item in text.split())
    kept = [item for item, count in counts.most_common(max_size) if count >= min_count]
    return {item: i for i, item in enumerate(kept)}


def load_train_df():
    """The notebook's train split (the test set is dropped and never used here)."""
    df = pd.read_pickle(ROOT / "embeddings_sampled.pkl")
    df = df[df[LABEL_COL].isin(orientation_to_side)].copy()
    df[LABEL_COL] = df[LABEL_COL].map(orientation_to_side)
    df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
    train_df, _test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[LABEL_COL])
    return train_df


def prepare_search_data(train_df, seed=SEED):
    """A GRID_SAMPLE-row sample of the train split, divided into train/validation, and the three encoder setups:
    {encoder name: (train pairs, validation pairs)}, plus the number of classes."""
    small = train_df.sample(n=GRID_SAMPLE, random_state=seed)
    tr_df, va_df = train_test_split(small, test_size=0.2, random_state=seed, stratify=small[LABEL_COL])
    label_names = sorted(train_df[LABEL_COL].unique())
    to_id = {name: i for i, name in enumerate(label_names)}

    def pairs(frame, column):
        texts = frame[column].str.join(" ") if column == "lemmas" else frame[column]
        return list(zip(texts, [to_id[name] for name in frame[LABEL_COL]]))

    vocab = get_vocab([text for text, _ in pairs(tr_df, "lemmas")], min_count=5, max_size=5000)
    setups = {
        "indicator": (TokenIndicator(vocab), pairs(tr_df, "lemmas"), pairs(va_df, "lemmas")),
        "bow": (BagOfWords(vocab), pairs(tr_df, "lemmas"), pairs(va_df, "lemmas")),
        "embeddings": (Embedding(small), pairs(tr_df, "sentence"), pairs(va_df, "sentence")),
    }
    return setups, len(label_names), va_df[LABEL_COL].value_counts(normalize=True).max()


def train_and_score(encoder, tr, va, num_classes, params, seed=SEED):
    """Trains one MLP setting (best epoch by validation loss) and returns its validation scores."""
    set_seed(seed)
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        metrics, best, preds = smlp.train_neural_network_model(
            tr, va, num_classes, encoder, save_to=str(MODEL_DIR / "model"), **params, **FIXED
        )
    return {"best_epoch": best + 1,
            "val_loss": metrics[best]["val_metrics"]["loss"],
            "val_accuracy": metrics[best]["val_metrics"]["accuracy"],
            "val_f1": f1_score([label for _, label in va], preds, average="macro"),
            "seconds": round(time.time() - t, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setups, num_classes, majority = prepare_search_data(load_train_df())
    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    _, tr, va = setups["indicator"]
    print(f"train {len(tr)} / val {len(va)}, majority baseline {majority:.4f}, "
          f"{len(grid)} settings x {len(setups)} encoders", flush=True)

    rows = []
    for name, (encoder, tr, va) in setups.items():
        for k, params in enumerate(grid):
            row = {"encoder": name, **params, **train_and_score(encoder, tr, va, num_classes, params)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
            print(f"[{name} {k + 1}/{len(grid)}] {params} -> acc {row['val_accuracy']:.4f} "
                  f"f1 {row['val_f1']:.4f} loss {row['val_loss']:.4f} epoch {row['best_epoch']} ({row['seconds']}s)", flush=True)

    results = pd.DataFrame(rows)
    print("\n=== best 5 per encoder (by validation macro F1) ===")
    for name, group in results.groupby("encoder", sort=False):
        print(f"\n{name}\n" + group.sort_values("val_f1", ascending=False).head(5).drop(columns="encoder").to_string(index=False))


if __name__ == "__main__":
    main()
