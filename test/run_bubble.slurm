#!/bin/bash
#SBATCH --job-name=bubble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=normal
#SBATCH --account=teorica
#SBATCH --constraint=matrix
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
# Una variante di bolla (test/integration_audit/bubble_audit.py) su uno stato fissato, poi pair_gamma con i
# flag HP.  Nessun ciclo: e' un test a un passo (2026-10-05, convergenza in p della bolla con Lambda = 8).
# Variabili:
#   VARIANT  nome in VARIANTS di bubble_audit.py (obbligatoria)
#   RUN IT   stato: default P0p90_t_frz 13 (cube e snap in seeds_warm, vedi sotto)
#   CUBES    default seeds_warm/P0p90_t_frz_it12
#   BASE     default seeds_warm/bubble_base (contiene RUN/snap/iterIT.npz)
#   QSET     default large
# Uscita: out/integration_audit/bubble_${RUN}_it${IT}_${QSET}/${VARIANT}/{impi,pair,pair_hp}
set -euo pipefail
cd "$SLURM_SUBMIT_DIR/.."
module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$SLURM_JOB_ID"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

: "${VARIANT:?serve VARIANT}"
RUN="${RUN:-P0p90_t_frz}"
IT="${IT:-13}"
CUBES="${CUBES:-seeds_warm/P0p90_t_frz_it12}"
BASE="${BASE:-seeds_warm/bubble_base}"
QSET="${QSET:-large}"
echo "== VARIANT=$VARIANT RUN=$RUN IT=$IT CUBES=$CUBES QSET=$QSET cpu=$SLURM_CPUS_PER_TASK =="

python3 test/integration_audit/bubble_audit.py compute --run "$RUN" --it "$IT" --base "$BASE" --cubes "$CUBES" \
    --qset "$QSET" --workers "$SLURM_CPUS_PER_TASK" --variants "$VARIANT"

D="out/integration_audit/bubble_${RUN}_it${IT}_${QSET}/${VARIANT}"
python3 -m fflo.pair_gamma --no-plots --impi-table-path "$D/impi/impi_table.npz" \
    --kk-pad 0 --kk-oversample 1 --kk-tail zero --kk-tail-alpha nan --kk-max-n 0 \
    --eta-gamma 0.002 --im-inv-sign-guard on --reference-mode proxy_residual \
    --qp-residual-mode cross_inc --qpqp-source auto --qpqp-xi-cutoff 0.015 --qpqp-n-p 5000 --qpqp-n-phi 24 \
    --residual-omega-taper-start 0 --residual-omega-taper-stop 10 --residual-omega-taper-mode qcut_cosine \
    --out-dir "$D/pair_hp" --ref-kk-fine-step 2.5e-4 --residual-taper-qfreeze 1 --residual-qmax-lambda 2 \
    > "$D/pair_hp.log" 2>&1
echo "== $VARIANT fatto: $D =="
