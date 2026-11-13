#!/bin/bash
# The Week 3 decoder test on a SLURM cluster: six action-vector sets, one GPU job each, run side by side.
#
#     sbatch -p <gpu partition> scripts/week3_decoder.sbatch      (from the repository folder, after `mkdir -p logs`)
#
# Each job trains the decoders for one action-vector set (2 state spaces x 2 decoders x 3 seeds) on training singles,
# stops them on validation singles, then scores validation singles and the 127 test doubles. Results go to
# results/week3_decoder/doubles__<tag>.csv, decoders to models/decoders/, logs to logs/week3_decoder_<n>.log.
# The six sets are the fixed list for this test (see README.md); do not add to it after looking at the scores.
#SBATCH --job-name=pertact-doubles
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/week3_decoder_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

TAGS=(main raw meanshift unit_infonce unit_meanshift anchor)
VARIANTS=(infonce_scaled_per_action infonce mean_shift infonce_unit mean_shift_unit infonce_anchor)
i=$SLURM_ARRAY_TASK_ID
REFS=()                                   # the Week 1 references (one-hot, PCA loadings) go with the first job only
if [ "$i" -eq 0 ]; then REFS=(onehot pca_loadings); fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# stop here if PyTorch cannot use the GPU, instead of running for hours on the CPU
python -I -c "import sys, torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"

# ${REFS[@]+...} expands to nothing for an empty list, also on old bash versions where "${REFS[@]}" would stop the script
python -I -m pertact.evaluate actions --doubles --variants "${VARIANTS[$i]}" --references ${REFS[@]+"${REFS[@]}"} --tag "${TAGS[$i]}"
