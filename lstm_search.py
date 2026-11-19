"""
Optuna hyperparameter search for the LSTM in lstm_experiment.py, followed by a re-evaluation of the best trials.

Phases (--phase):
    search   each trial trains on one fixed fold (--search-seed, --search-fold) of a training subsample (--frac,
             0.25 by default) and reports the validation metric after every epoch, so poor trials are pruned early.
    reeval   the --top-k best finished trials are trained again on --reeval-folds (folds not used for selection,
             so the ranking is not biased towards trials that got lucky on the search fold), via lstm_experiment.
    all      search, then reeval (default).

Several workers can share one study (the study is stored in a journal file), e.g. on one GPU:
    for w in 0 1 2 3; do python lstm_search.py --study-name s1 --n-workers 4 --worker-id $w & done; wait
The workers split the re-evaluation between them, and worker 0 writes the summary at the end.

Any option this script does not know is passed on to lstm_experiment.py, e.g. --epochs 15, --trainable, --device cuda.

Outputs, under <results-dir>/optuna/<study-name>/:
    journal.log                    the Optuna study (resumable; open with optuna.load_study)
    trials.csv                     every trial: parameters, value, state, best epoch, minutes
    trials/<key>/history.csv       per-epoch metrics of each trial
    trials/<key>/model.pkl         best weights of each completed trial (see lstm_experiment.load_model_pkl)
    importances.json               which parameters mattered most (Optuna's fANOVA estimate)
    reeval_summary.csv             top-k trials: mean ± std over the re-evaluation folds
    best_config.json               the winner, with the command line for the final runs
The re-evaluation runs themselves are normal lstm_experiment results folders, <study-name>_reeval_trial<N>.

Restarting: rerun the same command. Finished trials are kept; with --resume-interrupted, trials that were running
when the job died (state RUNNING) are marked failed and queued again, and continue from their epoch checkpoint.
Only use --resume-interrupted when no other worker of the study is still running.

Examples:
    python lstm_search.py --study-name smoke --n-trials 4 --top-k 2 --reeval-folds 1 2 --frac 0.01 --epochs 2
    python lstm_search.py --study-name lstm_s1 --n-trials 40 --timeout 14400 --wandb
"""

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_experiment as L  # noqa: E402


# ----------------------------------------------------------------------------
# Search space: edit here
# ----------------------------------------------------------------------------

def suggest(trial):
    return {
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "word_dropout": trial.suggest_float("word_dropout", 0.0, 0.3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "pooling": trial.suggest_categorical("pooling", ["last", "mean"]),
        "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
        # larger batches mainly speed things up on an A100; the lr range covers what they need
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


# how each searched parameter is passed to lstm_experiment.py
ARG_NAMES = {"hidden": "--hidden", "lr": "--lr", "weight_decay": "--weight-decay", "word_dropout": "--word-dropout",
             "dropout": "--dropout", "pooling": "--pooling", "batch_size": "--batch-size"}


def params_to_argv(params):
    argv = []
    for k, v in params.items():
        if k == "bidirectional":
            argv += ["--bidirectional"] if v else []
        else:
            argv += [ARG_NAMES[k], repr(v) if isinstance(v, float) else str(v)]
    return argv


def params_key(params):
    """Stable folder name for a parameter set, so a re-queued trial finds its own checkpoint."""
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study-name", default="lstm_search")
    p.add_argument("--phase", choices=["search", "reeval", "all", "final-args"], default="all",
                   help="final-args: print the lstm_experiment options of the --final-rank-th best re-evaluated "
                        "trial (0 = best), for the final runs")
    p.add_argument("--final-rank", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=40, help="total finished (complete or pruned) trials, all workers")
    p.add_argument("--timeout", type=float, default=None, help="stop starting new trials after this many seconds")
    p.add_argument("--metric", choices=["f1", "accuracy", "loss"], default="f1",
                   help="validation metric to optimise (macro F1 by default; the classes are 62/38)")
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--search-fold", type=int, default=0)
    p.add_argument("--reeval-folds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--startup-trials", type=int, default=10, help="random trials before TPE starts modelling")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--prune-startup", type=int, default=5, help="trials that are never pruned")
    p.add_argument("--prune-warmup", type=int, default=2, help="epochs before a trial can be pruned")
    p.add_argument("--resume-interrupted", action="store_true")
    p.add_argument("--wait-limit", type=float, default=7200,
                   help="seconds to wait for other workers before re-evaluating anyway")
    args, rest = p.parse_known_args(argv)
    if "--frac" not in rest:
        rest += ["--frac", "0.25"]
    exp = L.parse_args(rest + ["--name", "unused"])  # validates the forwarded options
    if not 0 <= args.worker_id < args.n_workers:
        p.error("--worker-id must be in [0, --n-workers)")
    return args, rest, exp


def load_study(args, study_dir, create=True):
    storage = JournalStorage(JournalFileBackend(str(study_dir / "journal.log")))
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.startup_trials, constant_liar=True,
                                         seed=1000 * args.search_seed + args.worker_id)
    pruner = (optuna.pruners.NopPruner() if args.no_prune else
              optuna.pruners.MedianPruner(n_startup_trials=args.prune_startup, n_warmup_steps=args.prune_warmup))
    direction = "minimize" if args.metric == "loss" else "maximize"
    return optuna.create_study(study_name=args.study_name, storage=storage, sampler=sampler, pruner=pruner,
                               direction=direction, load_if_exists=create)


def export_trials(study, study_dir):
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    tmp = study_dir / f"trials.csv.{os.getpid()}.tmp"  # one per process: workers export at the same time
    df.to_csv(tmp, index=False)
    tmp.replace(study_dir / "trials.csv")


def wandb_trial_run(exp, study_name, trial_number, tdir, cfg):
    import wandb
    id_file = tdir / "wandb_id.txt"
    run_id = id_file.read_text().strip() if id_file.exists() else __import__("secrets").token_hex(4)
    id_file.write_text(run_id)
    return wandb.init(project=exp.wandb_project, entity=exp.wandb_entity, group=study_name, job_type="search",
                      name=f"{study_name}-trial{trial_number}", id=run_id, resume="allow", dir=str(tdir),
                      mode=exp.wandb_mode, config=cfg | {"trial": trial_number})


def finished_count(study):
    return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))


def wait_for_search_end(study, args):
    """Wait until no trial of the study is running (other workers may still be busy)."""
    t0 = time.time()
    while True:
        running = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        if not running:
            return
        if time.time() - t0 > args.wait_limit:
            print(f"still {len(running)} running trials after {args.wait_limit:.0f}s; re-evaluating anyway",
                  flush=True)
            return
        time.sleep(30)


def top_trials(study, args):
    done = [t for t in study.get_trials(states=(TrialState.COMPLETE,)) if t.value is not None]
    sign = 1 if args.metric == "loss" else -1
    return sorted(done, key=lambda t: (sign * t.value, t.number))[:args.top_k]


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

def run_search(args, exp, study, study_dir):
    if args.resume_interrupted:
        for t in study.get_trials(states=(TrialState.RUNNING,)):
            # private API: Optuna has no public call to fail a trial that another (dead) process owned
            study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
            study.enqueue_trial(t.params, skip_if_exists=False)
            print(f"trial {t.number} was interrupted: queued again with the same parameters", flush=True)

    if finished_count(study) >= args.n_trials:
        print(f"search already has {finished_count(study)} finished trials", flush=True)
        return

    device = L.pick_device(exp.device)
    data = L.prepare_data(exp)
    tr_idx, va_idx = L.cross_validation_split(len(data["train_df"]), exp.folds, args.search_seed)[args.search_fold]
    ids, labels = data["train_ids"], data["train_labels"]
    train_set = L.LabelledSequences([ids[i] for i in tr_idx], [labels[i] for i in tr_idx])
    val_set = L.LabelledSequences([ids[i] for i in va_idx], [labels[i] for i in va_idx])
    base_cfg = vars(exp)
    print(f"worker {args.worker_id}: searching on seed {args.search_seed} fold {args.search_fold} "
          f"({len(train_set)} train / {len(val_set)} val) on {device}", flush=True)

    def objective(trial):
        params = suggest(trial)
        cfg = base_cfg | params
        tdir = study_dir / "trials" / params_key(params)
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "params.json").write_text(json.dumps(params, indent=2))
        print(f"\ntrial {trial.number}: {params}", flush=True)
        run = wandb_trial_run(exp, args.study_name, trial.number, tdir, cfg) if exp.wandb else None
        t0 = time.time()

        def on_epoch(epoch, row):
            if run:
                run.log({k: v for k, v in row.items() if k != "epoch"} | {"epoch": epoch}, step=epoch)
            trial.report(row[f"val_{args.metric}"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned after epoch {epoch}")

        try:
            model, history, best_epoch = L.train_one(cfg, data["emb"], data["pad_id"], len(data["label_names"]),
                                                     train_set, val_set, args.search_seed, device,
                                                     on_epoch=on_epoch, checkpoint_path=tdir / "checkpoint.pt")
        except optuna.TrialPruned:
            print(f"trial {trial.number}: pruned", flush=True)
            (tdir / "checkpoint.pt").unlink(missing_ok=True)
            if run:
                run.summary.update({"state": "pruned"})
                run.finish()
            raise
        except BaseException:
            if run:
                run.finish(exit_code=1)
            raise

        history.to_csv(tdir / "history.csv", index=False)
        va, *_ = L.predict(model, val_set, data["pad_id"], exp.eval_batch_size, device)
        attrs = {"best_epoch": best_epoch, "epochs_run": len(history), "minutes": (time.time() - t0) / 60,
                 "val_accuracy": va["accuracy"], "val_f1": va["f1"], "val_loss": va["loss"], "key": tdir.name}
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        if exp.save_models:
            L.save_model_pkl(tdir / "model.pkl", model, cfg, data["label_names"])
        (tdir / "checkpoint.pt").unlink(missing_ok=True)
        if run:
            run.summary.update(attrs | {"state": "complete"})
            run.finish()
        print(f"trial {trial.number}: val {args.metric} {va[args.metric]:.4f} "
              f"(best epoch {best_epoch}, {attrs['minutes']:.1f} min)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return va[args.metric]

    study.optimize(
        objective, timeout=args.timeout, gc_after_trial=True,
        callbacks=[optuna.study.MaxTrialsCallback(args.n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED)),
                   lambda s, t: export_trials(s, study_dir)],
    )


def run_reeval(args, rest, study, study_dir):
    wait_for_search_end(study, args)
    export_trials(study, study_dir)
    top = top_trials(study, args)
    if not top:
        raise SystemExit("no completed trials to re-evaluate")
    exp = L.parse_args(rest + ["--name", "unused"])
    names = {t.number: f"{args.study_name}_reeval_trial{t.number}" for t in top}

    # each worker re-evaluates its share of the top trials
    for rank, t in enumerate(top):
        if rank % args.n_workers != args.worker_id:
            continue
        print(f"\nre-evaluating trial {t.number} (rank {rank + 1}, search {args.metric} {t.value:.4f}) "
              f"on folds {args.reeval_folds}", flush=True)
        L.main(rest + params_to_argv(t.params) + [
            "--name", names[t.number], "--seeds", str(args.search_seed),
            "--only-folds", *map(str, args.reeval_folds),
        ])

    if args.worker_id != 0:
        return
    # worker 0 waits for the other workers' re-evaluations, then summarises
    t0 = time.time()
    while True:
        missing = [n for n, name in names.items() if not reeval_done(exp, name, args)]
        if not missing:
            break
        if time.time() - t0 > args.wait_limit:
            print(f"re-evaluation of trials {missing} not finished; summarising without them", flush=True)
            break
        time.sleep(30)
    summarise(args, rest, study, study_dir, top, names, exp)


def reeval_done(exp, name, args):
    path = Path(exp.results_dir) / name / "cv_scores.csv"
    if not path.exists():
        return False
    scores = pd.read_csv(path)
    return set(args.reeval_folds) <= set(scores[scores.seed == args.search_seed].fold)


def summarise(args, rest, study, study_dir, top, names, exp):
    rows = []
    for t in top:
        path = Path(exp.results_dir) / names[t.number] / "cv_scores.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s = s[(s.seed == args.search_seed) & s.fold.isin(args.reeval_folds)]
        rows.append({"trial": t.number, "search_value": t.value, "folds": len(s),
                     **{f"{m}_mean": s[m].mean() for m in ("accuracy", "f1", "loss")},
                     **{f"{m}_std": s[m].std() for m in ("accuracy", "f1", "loss")},
                     **t.params})
    summary = pd.DataFrame(rows).sort_values(f"{args.metric}_mean", ascending=args.metric == "loss")
    summary.to_csv(study_dir / "reeval_summary.csv", index=False)
    print("\nre-evaluation summary:")
    print(summary.round(4).to_string(index=False))

    best = summary.iloc[0]
    best_trial = next(t for t in top if t.number == best.trial)
    final_argv = params_to_argv(best_trial.params)
    (study_dir / "best_config.json").write_text(json.dumps({
        "trial": int(best.trial), "params": best_trial.params, "metric": args.metric,
        f"reeval_{args.metric}_mean": float(best[f"{args.metric}_mean"]),
        "final_command": "python lstm_experiment.py " + " ".join(final_argv)
                         + f" --name final_{args.study_name}_trial{int(best.trial)} --frac 1.0 --eval-test",
    }, indent=2))
    print(f"\nbest: trial {int(best.trial)} -> {study_dir / 'best_config.json'}")

    try:
        importances = optuna.importance.get_param_importances(study)
        (study_dir / "importances.json").write_text(json.dumps(importances, indent=2))
        print("parameter importances:", {k: round(v, 3) for k, v in importances.items()})
    except (RuntimeError, ValueError) as e:  # e.g. too few completed trials
        print(f"no parameter importances: {e}")


def print_final_args(args, study_dir):
    """Print options for lstm_experiment.py: the re-evaluated trial at --final-rank, with a matching --name."""
    summary_path = study_dir / "reeval_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found: run the search and re-evaluation first")
    summary = pd.read_csv(summary_path)  # already sorted best first
    if args.final_rank >= len(summary):
        raise SystemExit(f"only {len(summary)} re-evaluated trials")
    number = int(summary.iloc[args.final_rank].trial)
    study = optuna.load_study(study_name=args.study_name,
                              storage=JournalStorage(JournalFileBackend(str(study_dir / "journal.log"))))
    params = study.trials[number].params
    print(shlex.join(params_to_argv(params) + ["--name", f"final_{args.study_name}_trial{number}"]))


def main(argv=None):
    args, rest, exp = parse_args(argv)
    study_dir = Path(exp.results_dir) / "optuna" / args.study_name
    if args.phase == "final-args":
        print_final_args(args, study_dir)
        return
    study_dir.mkdir(parents=True, exist_ok=True)
    settings = study_dir / "settings.json"
    # settings that change results must match on restart; device, workers, W&B etc. may differ
    current = {"search": {k: v for k, v in vars(args).items()
                          if k in ("metric", "search_seed", "search_fold", "reeval_folds", "top_k")},
               "experiment": {k: v for k, v in vars(exp).items() if k not in L.RUN_ONLY_SETTINGS | {"name"}}}
    if settings.exists():
        saved = json.loads(settings.read_text())
        # older settings files may still contain run-only settings (e.g. paths): ignore those
        saved["experiment"] = {k: v for k, v in saved.get("experiment", {}).items()
                               if k not in L.RUN_ONLY_SETTINGS | {"name"}}
        if saved != current:
            raise SystemExit(f"{settings} has different settings; use another --study-name")
    settings.write_text(json.dumps(current, indent=2))

    study = load_study(args, study_dir)
    if args.phase in ("search", "all"):
        run_search(args, exp, study, study_dir)
        export_trials(study, study_dir)
    if args.phase in ("reeval", "all"):
        run_reeval(args, rest, study, study_dir)


if __name__ == "__main__":
    main()
