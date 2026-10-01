#!/bin/bash
# Scan di polarizzazione con Gamma piu' risolta del submit P originale.
#
# Uso sul cluster, dalla cartella test/:
#   ./submit_polarization_gfine.sh
#   ./submit_polarization_gfine.sh 0.675 0.725 0.75
#
# Variabili opzionali:
#   TARGET=45 ALPHA=0.20 PRUNE_KEEP=2 ./submit_polarization_gfine.sh ...
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TARGET="${TARGET:-45}"
ALPHA="${ALPHA:-0.20}"
PRUNE_KEEP="${PRUNE_KEEP:-2}"

if (( $# )); then
    POLS=("$@")
else
    # Anchor della vecchia scansione + punti nuovi attorno al cambio FFLO -> Q=0.
    POLS=(0.60 0.65 0.675 0.70 0.725 0.75 0.80)
fi

COMMON="--profile gold \
--omega-mode zero_and_thresholds \
--bubble-lambda 6 \
--p-nodes 46 \
--truncation-fix physical \
--taper-mode qkin_cosine \
--taper-stop 20 \
--im-sign-guard on \
--pintmax 12 \
--k-update-max 8 \
--eta-floor exact_zero \
--thouless-q-mode qff \
--high-k-sigma pair-contact \
--contact-tail \
--sigma-nomega 97 \
--alpha $ALPHA"

echo "Gamma-fine scan: P=${POLS[*]} TARGET=$TARGET ALPHA=$ALPHA"
for P in "${POLS[@]}"; do
    seed_tag="P${P/./p}"
    run_tag="${seed_tag}_gfine"

    python3 make_free_seed.py --P "$P" --out-dir "../seeds_free/$seed_tag"
    ARGS="$COMMON --seed-dir seeds_free/$seed_tag --prune-keep $PRUNE_KEEP"

    sbatch \
        --export=ALL,TAG="$run_tag",TARGET="$TARGET",ARGS="$ARGS" \
        --job-name="gf_${seed_tag}" \
        run.slurm
done
