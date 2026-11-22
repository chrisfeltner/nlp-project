from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Embed-and-pool": ROOT / "results" / "final_embed_pool_s1_trial19",
    "BiLSTM": ROOT / "results" / "final_lstm_s1_trial43",
}
FOLDS = (1, 2, 3)
LINESTYLES = {1: "-", 2: "--", 3: ":"}
COLORS = {"Embed-and-pool": "#16877A", "BiLSTM": "#D56A3A"}


def main():
    histories = {}
    missing = []
    for model_name, run_dir in RUNS.items():
        for fold in FOLDS:
            path = run_dir / "seed-0" / f"fold-{fold}" / "history.csv"
            if not path.is_file():
                missing.append(path)
            else:
                histories[(model_name, fold)] = pd.read_csv(path)
    if missing:
        paths = "\n".join(f"  {path.relative_to(ROOT)}" for path in missing)
        raise SystemExit(f"Missing per-epoch history files. Fetch these from Snellius first:\n{paths}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    metrics = (("train_loss", "Training loss"), ("val_loss", "Validation loss"))
    for model_name, _ in RUNS.items():
        for fold in FOLDS:
            history = histories[(model_name, fold)]
            for axis, (metric, _) in zip(axes, metrics):
                axis.plot(
                    history["epoch"], history[metric], color=COLORS[model_name],
                    linestyle=LINESTYLES[fold], linewidth=2,
                    label=f"{model_name}, fold {fold}",
                )

    for axis, (metric, title) in zip(axes, metrics):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.83))
    output = ROOT / "figures" / "slide4" / "embed_pool_vs_lstm_loss_seed0_folds1-3.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()