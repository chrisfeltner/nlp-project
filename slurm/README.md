# Running the LSTM experiments on the cluster

All commands are run by you (never by an agent: you are responsible for the SBUs). The project lives in
`~/nlp-project` on the cluster and the conda environment in `~/envs/nlp-project` (change `PROJECT_DIR` or
`ENV_PATH` in `env.sh` otherwise).
Submit every job from the project folder: Slurm writes the job log relative to where you run `sbatch`.

Costs: `gpu_a100` with 1 GPU + 18 cores + ≤120 GiB is one quarter node = **128 SBU per hour**; asking for more
cores or memory doubles it. `gpu_mig` (used only for the setup job) is 64 SBU per hour. Billing is for the
time the job runs, so keep `--time` close to what is needed (+20%) and `scancel` anything stuck.

Login nodes: only edit files, move data, submit and watch jobs. Anything over 15 minutes gets killed, and
there is no GPU, which is why the environment is built in a job.

## 1. Copy the code and the data

Create the folder on the cluster (`mkdir -p ~/nlp-project`) and copy into it, e.g. by dragging the files into
VS Code's remote Explorer, or with rsync from the laptop:

```bash
rsync -av --exclude .venv --exclude .git --exclude '/results' --exclude '/.cache' --exclude '/.hf_cache' \
      --exclude 'Pickles/merged_embeddings_sampled.pkl' --exclude 'part2/*.zip' \
      ./ snellius:nlp-project/
```

Needed on the cluster: `conda_environment.yaml`, `lstm_experiment.py`, `lstm_search.py`, `slurm/` (including
`slurm/logs/`, which must exist before the first `sbatch`: Slurm opens the log file before the job starts) and
`Pickles/merged_sampled_no_embeddings.pkl` (0.36 GB, made by `drop_embeddings.py`; not the 2.4 GB original).

## 2. One-time setup (a short job, ~30 min on gpu_mig, ≤32 SBU)

```bash
cd ~/nlp-project && sbatch slurm/setup_env.job
cat slurm/logs/lstm-setup-<jobid>.out
```

Builds the conda environment from `conda_environment.yaml` (or updates it, if you resubmit after changing the
file) and downloads HuBERT into `.hf_cache`. Check the CUDA lines in the output: `cuda available: True` and a
version for `built for cuda`. `None` there means a CPU-only PyTorch: fix that before any A100 job.

Then log in to W&B once (on the login node; it takes seconds), so jobs can upload live. The API key is in
your W&B user settings:

```bash
source activate ~/envs/nlp-project
wandb login
```

This stores the key in `~/.netrc` on the cluster (private to your account), which the jobs read.

## 3. Probe job (~20 min, ~43 credits)

```bash
sbatch slurm/probe.job
```

Read `slurm/logs/lstm-probe-<jobid>.out`: seconds per epoch at batch 64/256/512 and with 4 parallel
processes, plus GPU usage. Use it to pick `N_WORKERS` (search) and `N_PROCS` (final), and to check that the
`--time` limits fit.

## 4. Search job (limit 2.5 h, ≤320 SBU; expected ~1.25 h)

```bash
sbatch slurm/search.job                                   # defaults: STUDY=lstm_s1, 4 workers, 60 trials
sbatch --export=ALL,N_WORKERS=6,N_TRIALS=50 slurm/search.job
```

Writes `results/optuna/<study>/` (trials, `reeval_summary.csv`, `best_config.json`, importances).
If the job is killed or hits its time limit, submit it again: it continues the same study.

## 5. Final job (limit 2 h, ≤256 SBU; expected ~1–1.5 h)

```bash
sbatch slurm/final.job                                    # best configuration, 3 seeds x 5 folds
sbatch --dependency=afterok:<search jobid> slurm/final.job   # or: start right after the search
```

`FINAL_TOP=3` runs the top 3 configurations (3x the time). Finished runs are skipped on resubmission.

## Watching jobs and the budget

```bash
squeue -u $USER                  # your jobs; add --start for the expected start time
tail -f slurm/logs/lstm-search-<jobid>.out
scancel <jobid>                  # idle or stuck GPUs still cost SBUs
accinfo                          # budget and what is left
budget-overview -p gpu           # GPU budget at a glance
sacct -j <jobid> -o "JobID,JobName,MaxRSS,Elapsed,State"   # what a finished job used
```

After the probe, use `sacct` to check memory (`MaxRSS`) and set `--time` to the real duration + 20%.

## 6. Get the results back

```bash
# on the laptop
rsync -av snellius:nlp-project/results/ results/
```

W&B runs appear live in the `nlp-project-lstm` project while the jobs run. The probe job does not log to W&B.

If uploading ever fails (e.g. no internet on a node), submit with `--export=ALL,WANDB_MODE=offline` and upload
afterwards from the login node:

```bash
wandb sync results/optuna/lstm_s1/trials/*/wandb/offline-run-*
wandb sync results/*/seed-*/fold-*/wandb/offline-run-*
```
