#!/bin/bash
# PWIN (2026-10-02): ricetta PROD + finestre dense nella griglia in p del residuo della bolla.
#
# La parte residua A*A - A0*A0 ha una buca stretta a p = kF_up (non dipende da Q); i 31 nodi uniformi
# la perdono e -ImPi(qff, |Omega| < 0.01) viene sovrastimata di 2-3x: g_c a un passo +0.08 (P = 0.1) ..
# +0.02 (0.6), verificato con due metodi indipendenti (test/energy_bubble/eb_qff_scan.py).  Con
#   --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only   (31 -> 73 nodi in p)
# l'errore a un passo (con il taper della pipeline) scende a <= 0.006 per tutte le P da 0.10 a 0.75
# (test/energy_bubble/pwindow_probe.py; +-0.15 x 41 = 113 nodi e' simile ma costa 355 s per riga Q
# contro 204, 31 nodi: 61 s).
#
# Ogni P riparte dall'ultima iterazione completa di out/P0pXX_${SRC_SUFFIX} (stesso P: le cube si copiano).
# Al punto fisso lo spostamento di g_c e' amplificato (a P = 0.50 circa x3-5): e' quello che si misura.
#
# Uso (da test/):   DRY=1 ./submit_pwin.sh       POLS="0.30 0.50" TARGET=15 SRC_SUFFIX=prod
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLS=(${POLS:-0.30 0.50})
TARGET="${TARGET:-15}"
SRC_SUFFIX="${SRC_SUFFIX:-prod}"
SUFFIX="${SUFFIX:-pwin}"
DRY="${DRY:-0}"
ARGS_BASE="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --sigma-nomega 97 --high-k-sigma pair-contact --contact-tail --prune-keep 2 --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only"

for P in "${POLS[@]}"; do
  tag="P${P/./p}_${SUFFIX}"
  src="../out/P${P/./p}_${SRC_SUFFIX}"
  if [ -d "../out/$tag" ]; then
    echo "  $tag: out/$tag esiste gia', salto (la sua catena riprende da sola)"
    continue
  fi
  read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('$src')
assert paths, 'nessuna iterazione completa in $src'
print(last, paths[0], paths[1])")
  seed="../seeds_warm/$tag"
  args="$ARGS_BASE --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
  case "$args" in
    *,*) echo "ERRORE: virgola in ARGS: sbatch --export la usa come separatore"; exit 1 ;;
  esac
  cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TARGET",ARGS="$args" --job-name="slim_$tag" run.slurm)
  if [ "$DRY" = 1 ]; then
    echo "  DRY: $tag da $(basename "$src") it $it; ${cmd[*]}"
    continue
  fi
  mkdir -p "$seed"
  cp "$up" "$seed/A_komega_spinup_reseed.npz"
  cp "$down" "$seed/A_komega_spindown_reseed.npz"
  env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
  echo "  $tag: P = $P da $(basename "$src") it $it"
done
