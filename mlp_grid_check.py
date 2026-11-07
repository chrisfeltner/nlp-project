"""Checks the best settings from mlp_grid_search.py (plus the notebook's current setting) on several seeds.

Each seed draws its own search sample and train/validation split and its own initialisation, so this shows
whether a setting's advantage holds up beyond one split. Results go to grid_search/check.csv.

Run from the repo root after mlp_grid_search.py: python mlp_grid_check.py
"""
import pandas as pd

from mlp_grid_search import GRID, MODEL_DIR, OUT, load_train_df, prepare_search_data, train_and_score

SEEDS = [0, 1, 2]
TOP_K = 3
# the notebook's current 2.A-2.C settings, as a reference
CURRENT = {
    "indicator": {"hidden_dim": 128, "dropout": 0.1, "learning_rate": 1e-4, "batch_size": 512},
    "bow": {"hidden_dim": 128, "dropout": 0.1, "learning_rate": 1e-4, "batch_size": 512},
    "embeddings": {"hidden_dim": 128, "dropout": 0.1, "learning_rate": 1e-3, "batch_size": 512},
}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(OUT / "results.csv")
    candidates = {}
    for name, group in grid.groupby("encoder", sort=False):
        top = group.sort_values("val_f1", ascending=False).head(TOP_K)[list(GRID)].to_dict("records")
        candidates[name] = top + ([CURRENT[name]] if CURRENT[name] not in top else [])

    train_df = load_train_df()
    rows = []
    for seed in SEEDS:
        setups, num_classes, _ = prepare_search_data(train_df, seed=seed)
        for name, settings in candidates.items():
            encoder, tr, va = setups[name]
            for params in settings:
                row = {"encoder": name, "seed": seed, **params, "current": params == CURRENT[name],
                       **train_and_score(encoder, tr, va, num_classes, params, seed=seed)}
                rows.append(row)
                pd.DataFrame(rows).to_csv(OUT / "check.csv", index=False)
                print(f"[seed {seed} {name}] {params} -> f1 {row['val_f1']:.4f} acc {row['val_accuracy']:.4f} "
                      f"epoch {row['best_epoch']} ({row['seconds']}s)", flush=True)

    results = pd.DataFrame(rows)
    summary = (results.groupby(["encoder", *GRID, "current"], sort=False)
               .agg(f1_mean=("val_f1", "mean"), f1_std=("val_f1", "std"),
                    acc_mean=("val_accuracy", "mean"), seconds=("seconds", "mean"))
               .reset_index().sort_values(["encoder", "f1_mean"], ascending=[True, False]))
    print("\n=== mean over seeds ===\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
