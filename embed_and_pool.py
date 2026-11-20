"""
Embed-and-pool sentence classifier (Left vs Right) on frozen or trainable HuBERT input embeddings.

Self-contained: data loading, tokenization, model, training loop, cross-validation and logging
are all in this file (it does not use part2/clslib.py).

The data split matches Project 1, so cross-validation scores can be paired with Project 1 runs:
    - 90/10 train/test split with random_state=42, stratified on the label
    - for each seed: shuffle train rows with random.seed(seed) and np.array_split into k folds
    - the seed also sets model initialisation and batch order

Examples (run from the repository root):
    # quick end-to-end check on 1% of the data
    python embed_and_pool.py --smoke

    # one configuration, 3 seeds x 5 folds, on all training rows
    python embed_and_pool.py --name mean_pool

    # search-style run: 25% of the training rows, first fold of seed 0 only
    python embed_and_pool.py --name search_mean --frac 0.25 --seeds 0 --only-folds 0

Outputs, under <results-dir>/<name>/:
    config.json                                  every setting of this experiment
    cv_scores.csv                                one row per finished (seed, fold)
    test_scores.csv                              same, on the held-out test set (with --eval-test)
    seed-<s>/fold-<f>/history.csv                train/val loss, accuracy and macro F1 per epoch
    seed-<s>/fold-<f>/val_predictions.csv        per validation sentence: row id, gold, predicted, log-probs
    seed-<s>/fold-<f>/test_predictions.csv       the same for the test set (with --eval-test)
    seed-<s>/fold-<f>/model.pkl                  best weights, config and labels (load with load_model_pkl;
                                                 a frozen embedding table is left out and reloaded from HuBERT)
    seed-<s>/fold-<f>/checkpoint.pt              full training state after the last finished epoch (deleted when done)

Restarting: rerun the same command. A (seed, fold) that already has val_predictions.csv is skipped,
and an unfinished one continues from its checkpoint.pt after the last finished epoch.

Weights & Biases (optional): add --wandb to log every (seed, fold) as a run, grouped by --name.
On compute nodes without internet, add --wandb-mode offline and run `wandb sync` on the run folders later.
"""

import argparse
import fcntl
import hashlib
import json
import pickle
import random
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
LABEL_COL = "Party_orientation"
TEXT_COL = "sentence"  # original casing; HuBERT is case-sensitive


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_data(path, sample_frac=1.0):
    """Load the dataset and make the same train/test split as Project 1."""
    df = pd.read_pickle(path)
    df = df[[TEXT_COL, LABEL_COL]]  # only the columns this script needs
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[LABEL_COL])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def cross_validation_split(n, k, seed):
    """Same procedure as Project 1's part1.experiment.cross_validation_split, on row positions 0..n-1.
    Returns a list of k (train_ids, val_ids) pairs."""
    random.seed(seed)
    ids = list(range(n))
    random.shuffle(ids)
    parts = np.array_split(np.array(ids), k)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != i]), parts[i]) for i in range(k)]


def tokenize(texts, tokenizer, max_len, cache_dir, model_name):
    """Token ids for each text (no [CLS]/[SEP]), truncated to max_len, cached on disk."""
    key = hashlib.sha1(("\n".join(texts) + f"|{model_name}|{max_len}").encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"ids_{key}.pt"
    if cache.exists():
        return torch.load(cache)
    ids = []
    for i in range(0, len(texts), 10_000):
        enc = tokenizer(texts[i:i + 10_000], add_special_tokens=False, truncation=True, max_length=max_len)
        ids.extend(enc["input_ids"])
    ids = [x if x else [tokenizer.unk_token_id] for x in ids]  # never return an empty sequence
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache)
    return ids


class LabelledSequences(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long), self.labels[i]


def make_collate(pad_id):
    def collate(batch):
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        x = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        return x, lengths, torch.tensor(labels, dtype=torch.long)
    return collate


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def hubert_embedding_matrix(model_name):
    """HuBERT's input embedding table (vocab_size x 768), with the [PAD] row set to zero."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    weights = AutoModel.from_pretrained(model_name).get_input_embeddings().weight.detach().clone()
    weights[tokenizer.pad_token_id] = 0.0
    return tokenizer, weights


class EmbedAndPoolClassifier(nn.Module):
    """Masked token embeddings -> mean/sum pooling -> MLP head -> log-probabilities."""

    def __init__(self, embedding_matrix, pad_id, num_classes, pooling="mean", dropout=0.0,
                 head_sizes=(50, 50), freeze_embeddings=True):
        super().__init__()
        self.pad_id = pad_id
        self.pooling = pooling
        # Keep each fold's trainable embedding table independent of the shared HuBERT weights.
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix.clone(), freeze=freeze_embeddings,
                                                      padding_idx=pad_id)
        layers, in_dim = [], embedding_matrix.shape[1]
        for size in head_sizes:
            layers += [nn.Dropout(dropout), nn.Linear(in_dim, size), nn.Tanh()]
            in_dim = size
        layers += [nn.Dropout(dropout), nn.Linear(in_dim, num_classes), nn.LogSoftmax(dim=-1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        e = self.embedding(x)
        valid = x.ne(self.pad_id)
        h = (e * valid.unsqueeze(-1)).sum(dim=1)
        if self.pooling == "mean":
            h = h / valid.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(h)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def trainable_state(model, freeze):
    """Model weights on the CPU, without the embedding table when it is frozen (it never changes)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not (freeze and k.startswith("embedding."))}


def load_trainable_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("embedding.") for k in missing):
        raise RuntimeError(f"state does not fit the model: missing {missing}, unexpected {unexpected}")


def save_model_pkl(path, model, cfg, label_names):
    """Pickle the model's weights (as numpy arrays), its config and the label names.
    A frozen embedding table is left out; load_model_pkl takes it from HuBERT again."""
    obj = {
        "format": "lstm_experiment/1",
        "config": dict(cfg),
        "label_names": list(label_names),
        "pad_id": model.pad_id,
        "state_dict": {k: v.numpy() for k, v in trainable_state(model, cfg["freeze"]).items()},
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_model_pkl(path, device="cpu", embedding_matrix=None):
    """Rebuild a model saved by save_model_pkl. Returns (model in eval mode, saved dict).
    embedding_matrix: HuBERT's table, if already loaded (otherwise it is loaded from the saved model_name)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    cfg = obj["config"]
    if embedding_matrix is None:
        _, embedding_matrix = hubert_embedding_matrix(cfg["model_name"])
    model = EmbedAndPoolClassifier(
        embedding_matrix, obj["pad_id"], len(obj["label_names"]),
        pooling=cfg["pooling"], dropout=cfg["dropout"],
        head_sizes=cfg["head_sizes"], freeze_embeddings=cfg["freeze"],
    )
    load_trainable_state(model, {k: torch.from_numpy(v) for k, v in obj["state_dict"].items()})
    return model.to(device).eval(), obj


def save_atomic(obj, path):
    """Write to a temporary file first, so a crash mid-save never leaves a broken checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, criterion, optimizer=None, clip=0.0):
    """One pass over loader; trains if an optimizer is given. Returns metrics and per-example outputs."""
    training = optimizer is not None
    model.train(training)
    total_loss, golds, preds, logprobs = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, lengths)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * len(y)
            golds.append(y.cpu())
            logprobs.append(out.detach().float().cpu())
    golds = torch.cat(golds).numpy()
    logprobs = torch.cat(logprobs).numpy()
    preds = logprobs.argmax(axis=1)
    return {
        "loss": total_loss / len(golds),
        "accuracy": accuracy_score(golds, preds),
        "f1": f1_score(golds, preds, average="macro"),
    }, golds, preds, logprobs


def train_one(cfg, embedding_matrix, pad_id, num_classes, train_set, val_set, seed, device, on_epoch=None,
              checkpoint_path=None):
    """
    Train one model with early stopping on validation loss and return the best model and its history.

    on_epoch(epoch, row) is called after every epoch with that epoch's history row (train_*/val_* metrics),
    e.g. for logging or Optuna pruning; it may raise to stop training.
    checkpoint_path: if given, the full training state is saved there after every epoch, and training
    resumes from it when the file already exists (so a killed run continues from its last finished epoch).
    """
    set_seed(seed)
    collate = make_collate(pad_id)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
                              generator=gen, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["eval_batch_size"], shuffle=False, collate_fn=collate,
                            num_workers=cfg["num_workers"])

    model = EmbedAndPoolClassifier(
        embedding_matrix, pad_id, num_classes,
        pooling=cfg["pooling"], dropout=cfg["dropout"],
        head_sizes=cfg["head_sizes"], freeze_embeddings=cfg["freeze"],
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.NLLLoss()

    history, best_loss, best_state, best_epoch, bad_epochs, start_epoch = [], float("inf"), None, -1, 0, 0
    if checkpoint_path is not None and checkpoint_path.exists():
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # our own file
        load_trainable_state(model, ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        history, best_loss, best_state = ck["history"], ck["best_loss"], ck["best_state"]
        best_epoch, bad_epochs, start_epoch = ck["best_epoch"], ck["bad_epochs"], ck["epoch"] + 1
        gen.set_state(ck["loader_rng"])
        set_rng_state(ck["rng"])
        print(f"  resumed from checkpoint after epoch {ck['epoch']}", flush=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        if bad_epochs >= cfg["patience"]:  # only when resuming a run that had already stopped
            break
        t0 = time.time()
        tr, *_ = run_epoch(model, train_loader, device, criterion, optimizer, clip=cfg["clip"])
        va, *_ = run_epoch(model, val_loader, device, criterion)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, "seconds": time.time() - t0}
        history.append(row)
        print(f"  epoch {epoch:02d} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} f1 {va['f1']:.4f} | {time.time() - t0:.0f}s",
              flush=True)
        if va["loss"] < best_loss:
            best_loss, best_epoch, bad_epochs = va["loss"], epoch, 0
            best_state = trainable_state(model, cfg["freeze"])
        else:
            bad_epochs += 1
        if checkpoint_path is not None:
            save_atomic({
                "epoch": epoch, "model": trainable_state(model, cfg["freeze"]), "optimizer": optimizer.state_dict(),
                "history": history, "best_loss": best_loss, "best_state": best_state, "best_epoch": best_epoch,
                "bad_epochs": bad_epochs, "loader_rng": gen.get_state(), "rng": get_rng_state(),
            }, checkpoint_path)
        if on_epoch is not None:
            on_epoch(epoch, row)
        if bad_epochs >= cfg["patience"]:
            print(f"  early stop: no val-loss improvement for {cfg['patience']} epochs", flush=True)
            break

    load_trainable_state(model, best_state)
    return model, pd.DataFrame(history), best_epoch


class WandbRun:
    """One Weights & Biases run per (seed, fold), grouped by experiment name; resumes the same W&B run on restart."""

    def __init__(self, args, cfg, run_dir, seed, fold):
        import wandb  # only needed with --wandb
        id_file = run_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
        id_file.write_text(run_id)
        self.run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.name, job_type="cv",
            name=f"{args.name}-seed{seed}-fold{fold}", id=run_id, resume="allow", dir=str(run_dir),
            mode=args.wandb_mode, config={**cfg, "seed": seed, "fold": fold},
        )

    def log_epoch(self, epoch, row):
        self.run.log({k: v for k, v in row.items() if k != "epoch"} | {"epoch": epoch}, step=epoch)

    def finish(self, summary):
        self.run.summary.update(summary)
        self.run.finish()


def predict(model, dataset, pad_id, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id))
    metrics, golds, preds, logprobs = run_epoch(model, loader, device, nn.NLLLoss())
    return metrics, golds, preds, logprobs


def predictions_frame(row_ids, golds, preds, logprobs, label_names):
    frame = pd.DataFrame({"row_id": row_ids, "gold": [label_names[g] for g in golds],
                          "pred": [label_names[p] for p in preds]})
    for j, name in enumerate(label_names):
        frame[f"logprob_{name}"] = logprobs[:, j]
    return frame


def append_row(path, row):
    """Append one row to a CSV; locked, so parallel processes of one experiment can share the file."""
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            pd.DataFrame([row]).to_csv(f, header=f.tell() == 0, index=False)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("experiment")
    g.add_argument("--name", default=None, help="results subfolder (default: built from the settings)")
    g.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    g.add_argument("--data", default=str(REPO_ROOT / "Pickles" / "merged_sampled_no_embeddings.pkl"),
                   help="dataset pickle (made by drop_embeddings.py; the full pickle also works)")
    g.add_argument("--cache-dir", default=str(REPO_ROOT / ".cache"))
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    g.add_argument("--only-folds", type=int, nargs="+", default=None, help="run only these fold indices")
    g.add_argument("--frac", type=float, default=1.0,
                   help="fraction of training rows to use (sampled after the train/test split; test set unchanged)")
    g.add_argument("--eval-test", action="store_true", help="also score each fold's model on the test set")
    g.add_argument("--no-save-models", dest="save_models", action="store_false",
                   help="do not write model.pkl (the best weights) for each finished run")
    g.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end check: 1%% of training rows, seed 0, 2 folds, 2 epochs, test eval on")
    g.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    g.add_argument("--num-workers", type=int, default=0)

    w = p.add_argument_group("weights & biases (optional, needs `pip install wandb` and `wandb login`)")
    w.add_argument("--wandb", action="store_true", help="log every (seed, fold) as a W&B run")
    w.add_argument("--wandb-project", default="nlp-project-lstm")
    w.add_argument("--wandb-entity", default=None, help="W&B team or user (default: your default entity)")
    w.add_argument("--wandb-mode", choices=["online", "offline"], default=None,
                   help="offline writes locally for a later `wandb sync` (for compute nodes without internet)")

    m = p.add_argument_group("model")
    m.add_argument("--model-name", default="SZTAKI-HLT/hubert-base-cc")
    m.add_argument("--max-len", type=int, default=256, help="truncate sentences to this many subword pieces")
    m.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    m.add_argument("--dropout", type=float, default=0.0, help="dropout in the MLP head")
    m.add_argument("--head-sizes", type=int, nargs="*", default=[50, 50])
    m.add_argument("--trainable", dest="freeze", action="store_false", help="fine-tune the embedding table")

    o = p.add_argument_group("optimisation")
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--weight-decay", type=float, default=1e-4)
    o.add_argument("--batch-size", type=int, default=64)
    o.add_argument("--eval-batch-size", type=int, default=512)
    o.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
    o.add_argument("--patience", type=int, default=3, help="early stopping patience on validation loss")
    o.add_argument("--clip", type=float, default=1.0, help="gradient norm clipping (0 to disable)")

    args = p.parse_args(argv)
    if args.smoke:
        args.frac, args.seeds, args.folds, args.epochs, args.eval_test = 0.01, [0], 2, 2, True
        args.name = args.name or "smoke"
    if args.name is None:
        args.name = (f"embed_and_pool_{args.pooling}_do{args.dropout}_lr{args.lr}_"
                     f"{'frozen' if args.freeze else 'trainable'}"
                     f"_frac{args.frac}")
    return args


def prepare_data(args):
    """Load the split, subsample training rows (--frac), tokenize, and load HuBERT's embedding table."""
    t0 = time.time()
    train_df, test_df = load_data(args.data)
    label_names = sorted(train_df[LABEL_COL].unique())
    label_to_id = {name: i for i, name in enumerate(label_names)}
    if args.frac < 1.0:
        train_df = train_df.sample(frac=args.frac, random_state=42).reset_index(drop=True)
    print(f"data: {len(train_df)} train rows, {len(test_df)} test rows, labels {label_names} "
          f"({time.time() - t0:.0f}s)", flush=True)

    tokenizer, emb = hubert_embedding_matrix(args.model_name)
    t0 = time.time()
    train_ids = tokenize(train_df[TEXT_COL].tolist(), tokenizer, args.max_len, args.cache_dir, args.model_name)
    test_set = None
    if args.eval_test:
        test_ids = tokenize(test_df[TEXT_COL].tolist(), tokenizer, args.max_len, args.cache_dir, args.model_name)
        test_set = LabelledSequences(test_ids, test_df[LABEL_COL].map(label_to_id).tolist())
    print(f"tokenized ({time.time() - t0:.0f}s), mean length "
          f"{np.mean([len(x) for x in train_ids]):.1f} pieces", flush=True)
    return {
        "train_df": train_df, "test_df": test_df, "label_names": label_names,
        "train_ids": train_ids, "train_labels": train_df[LABEL_COL].map(label_to_id).tolist(),
        "test_set": test_set, "emb": emb, "pad_id": tokenizer.pad_token_id,
    }


# settings that may differ between runs of the same experiment (they do not change any result);
# the paths are here so results stay usable after moving the repository or copying them to/from the cluster
RUN_ONLY_SETTINGS = {"seeds", "only_folds", "eval_test", "save_models", "device", "num_workers", "eval_batch_size",
                     "wandb", "wandb_project", "wandb_entity", "wandb_mode", "results_dir", "cache_dir", "data"}


def main(argv=None):
    args = parse_args(argv)
    cfg = vars(args)
    out_dir = Path(args.results_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    if config_path.exists():
        old = {k: v for k, v in json.loads(config_path.read_text()).items() if k not in RUN_ONLY_SETTINGS}
        new = {k: v for k, v in cfg.items() if k not in RUN_ONLY_SETTINGS}
        if old != new:
            raise SystemExit(f"{config_path} has different settings; use another --name or delete that folder")
    config_path.write_text(json.dumps(cfg, indent=2))
    device = pick_device(args.device)
    print(f"experiment {args.name} on {device}", flush=True)

    data = prepare_data(args)
    train_df, test_df, label_names = data["train_df"], data["test_df"], data["label_names"]
    train_ids, train_labels, emb, pad_id = data["train_ids"], data["train_labels"], data["emb"], data["pad_id"]
    test_set = data["test_set"]

    folds_to_run = args.only_folds if args.only_folds is not None else range(args.folds)
    for seed in args.seeds:
        splits = cross_validation_split(len(train_df), args.folds, seed)
        for fold in folds_to_run:
            run_dir = out_dir / f"seed-{seed}" / f"fold-{fold}"
            if (run_dir / "val_predictions.csv").exists():
                print(f"seed {seed} fold {fold}: already done, skipping", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            tr_idx, va_idx = splits[fold]
            train_set = LabelledSequences([train_ids[i] for i in tr_idx], [train_labels[i] for i in tr_idx])
            val_set = LabelledSequences([train_ids[i] for i in va_idx], [train_labels[i] for i in va_idx])
            print(f"seed {seed} fold {fold}: {len(train_set)} train / {len(val_set)} val", flush=True)

            t0 = time.time()
            wb = WandbRun(args, cfg, run_dir, seed, fold) if args.wandb else None
            checkpoint = run_dir / "checkpoint.pt"
            try:
                model, history, best_epoch = train_one(cfg, emb, pad_id, len(label_names), train_set, val_set,
                                                       seed, device, on_epoch=wb.log_epoch if wb else None,
                                                       checkpoint_path=checkpoint)
            except BaseException:
                if wb:  # close the W&B run as failed; a restart resumes it from the checkpoint
                    wb.run.finish(exit_code=1)
                raise
            history.to_csv(run_dir / "history.csv", index=False)
            summary = {"best_epoch": best_epoch}
            if args.save_models:
                save_model_pkl(run_dir / "model.pkl", model, cfg, label_names)

            if args.eval_test:
                te, golds, preds, logprobs = predict(model, test_set, pad_id, args.eval_batch_size, device)
                predictions_frame(np.arange(len(test_df)), golds, preds, logprobs, label_names).to_csv(
                    run_dir / "test_predictions.csv", index=False)
                append_row(out_dir / "test_scores.csv", {"seed": seed, "fold": fold, "accuracy": te["accuracy"],
                                                         "f1": te["f1"]})
                summary |= {"test_accuracy": te["accuracy"], "test_f1": te["f1"]}
            va, golds, preds, logprobs = predict(model, val_set, pad_id, args.eval_batch_size, device)
            summary |= {"best_val_accuracy": va["accuracy"], "best_val_f1": va["f1"], "best_val_loss": va["loss"]}
            append_row(out_dir / "cv_scores.csv", {
                "seed": seed, "fold": fold, "accuracy": va["accuracy"], "f1": va["f1"], "loss": va["loss"],
                "best_epoch": best_epoch, "epochs_run": len(history), "minutes": (time.time() - t0) / 60,
            })
            # written last: its presence marks this (seed, fold) as finished
            predictions_frame(va_idx, golds, preds, logprobs, label_names).to_csv(
                run_dir / "val_predictions.csv", index=False)
            checkpoint.unlink(missing_ok=True)  # the run is finished, its checkpoint is no longer needed
            if wb:
                wb.finish(summary)
            print(f"seed {seed} fold {fold}: val acc {va['accuracy']:.4f} f1 {va['f1']:.4f} "
                  f"(best epoch {best_epoch}, {(time.time() - t0) / 60:.1f} min)", flush=True)

    scores = pd.read_csv(out_dir / "cv_scores.csv")
    print(f"\n{args.name}: {len(scores)} runs, val accuracy {scores.accuracy.mean():.4f} "
          f"± {scores.accuracy.std():.4f}, macro F1 {scores.f1.mean():.4f} ± {scores.f1.std():.4f}")


if __name__ == "__main__":
    main()
