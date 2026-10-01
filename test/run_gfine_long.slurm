#!/bin/bash
#SBATCH --job-name=fflo_gold
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --account=teorica
#SBATCH --constraint=matrix
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --mail-user=francesco.pirolo2@unibo.it
#SBATCH --mail-type=FAIL

# Runner lungo per lo scan di polarizzazione con profile=gold. Il runner
# standard da 4 h non completa una singola iterazione gold. Il budget di
# 690 min lascia mezz'ora a Slurm e permette l'autochain fra iterazioni complete.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$SLURM_JOB_ID"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

: "${TAG:?serve TAG}" "${TARGET:?serve TARGET}"
ARGS="${ARGS:-}"
echo "== TAG=$TAG TARGET=$TARGET cpu=$SLURM_CPUS_PER_TASK ARGS=$ARGS =="

# shellcheck disable=SC2086
python3 test/run_test.py --out "out/$TAG" --target "$TARGET" --iters 1000 \
    --workers "$SLURM_CPUS_PER_TASK" --time-budget 690 $ARGS

DONE=$(python3 -c "import sys; sys.path.insert(0, 'test'); import run_test; print(run_test.last_complete_iteration('out/$TAG')[0])")
echo "== $TAG: $DONE / $TARGET iterazioni complete =="
if [ "${AUTOCHAIN:-1}" = "1" ] && [ "$DONE" -lt "$TARGET" ]; then
  echo "== $TAG: richiedo la continuazione gold =="
  cd "$SLURM_SUBMIT_DIR"
  sbatch --export=ALL,TAG="$TAG",TARGET="$TARGET",ARGS="$ARGS" \
         --job-name="gold_$TAG" run_gfine_long.slurm
fi
