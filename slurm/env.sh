# Shared settings for the jobs and setup_env.job: `source slurm/env.sh` from the project directory.
# Adjust PROJECT_DIR / ENV_PATH if your copy of the repository or the environment lives elsewhere.

export PROJECT_DIR="${PROJECT_DIR:-$HOME/nlp-project}"
export ENV_PATH="${ENV_PATH:-$HOME/envs/nlp-project}"

module purge
module load 2025
module load Anaconda3/2025.06-1

# HuBERT is downloaded once by setup_env.job into this cache; jobs then never need the internet for it
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TOKENIZERS_PARALLELISM=false
# W&B uploads live from the compute node (needs a one-time `wandb login` on the cluster, see README.md).
# If uploads ever fail, submit with `sbatch --export=ALL,WANDB_MODE=offline ...` and `wandb sync` afterwards.
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1   # like `python -u`: logs show up live in the job output

mkdir -p "$PROJECT_DIR/slurm/logs"

cd "$PROJECT_DIR"
