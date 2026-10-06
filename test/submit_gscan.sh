#!/bin/bash
# GSCAN (2026-10-02): alta P a accoppiamento FISSATO invece del ri-pin a ogni iterazione.
#
# Ad alta P ReGamma^-1(Q, 0) e' quasi piatta su tutto il disco Q < qff: il ri-pin a ogni iterazione
# (g ricalcolato dallo stato) rende il ciclo mal condizionato, con transitori di decine di iterazioni.
# A g fisso e SOTTO la criticita' il ciclo e' un normale problema autoconsistente.  Per ogni P si girano
# 2-3 valori di g; dallo stato convergito di ciascuno test/gscan_report.py calcola
#     g_c(stato) = -4 pi max[ReG^-1(qff, 0), ReG^-1(Q0+, 0)] - ln2/2     (due candidati, come qff-or-zero)
#     h(g) = g_c(stato) - g          > 0: sotto la criticita'
# e g_c e' dove h(g) = 0 (interpolazione).  Se a un g il canale supera la criticita' (h < 0), il ciclo
# puo' scappare: e' comunque un'informazione (g_c sta sotto quel g).
# shift = -(g + ln2/2) / (4 pi)  passato a run_test --pair-shift-fixed (density aggiunge delta = 1e-3).
#
# Ricetta = quella di submit_final.sh (tutte le correzioni), piu' --pair-shift-fixed.
# Sorgente: l'ultima iterazione completa di out/P0pXX_${SRC_SUFFIX} (default final), altrimenti di
# out/P0pXX_${SRC_FALLBACK}.  Tag P0pXX_gs<g>, con g scritto come p0p30 (+0.30) o m0p10 (-0.10).
#
# Uso (da test/):   POINTS="0.80:0.22/0.26/0.30 0.90:0.45/0.50/0.55" DRY=1 ./submit_gscan.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POINTS="${POINTS:?serve POINTS, per esempio POINTS=\"0.80:0.22/0.26/0.30\"}"
TARGET="${TARGET:-20}"
SRC_SUFFIX="${SRC_SUFFIX:-final}"
SRC_FALLBACK="${SRC_FALLBACK:-x75a0p3}"
DRY="${DRY:-0}"
ARGS_BASE="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --high-k-sigma pair-contact --contact-tail --prune-keep 2 --thouless-q-mode qff-or-zero --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only --lattice-dw 2.5e-4 --angular-feature-nodes 21 --sigma-nomega 193 --sigma-omega-dense-w 0.06 --sigma-omega-dense-stride 2 --sigma-k-kf-offsets 0.03:0.1:0.3:0.6 --sigma-nk 20"

for item in $POINTS; do
  P="${item%%:*}"
  gl="${item#*:}"
  src="../out/P${P/./p}_${SRC_SUFFIX}"
  python3 -c "import run_test, sys; sys.exit(0 if run_test.last_complete_iteration('$src')[1] else 1)" \
    || src="../out/P${P/./p}_${SRC_FALLBACK}"
  read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('$src')
assert paths, 'nessuna iterazione completa in $src'
print(last, paths[0], paths[1])")
  for g in ${gl//\// }; do
    gtag=$(python3 -c "g = float('$g'); print(('p' if g >= 0 else 'm') + f'{abs(g):.2f}'.replace('.', 'p'))")
    shift=$(python3 -c "import math; print(f'{-(float(\"$g\") + 0.5 * math.log(2.0)) / (4.0 * math.pi):.12f}')")
    tag="P${P/./p}_gs${gtag}"
    if [ -d "../out/$tag" ]; then
      echo "  $tag: out/$tag esiste gia', salto"
      continue
    fi
    seed="../seeds_warm/$tag"
    args="$ARGS_BASE --pair-shift-fixed $shift --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
    case "$args" in
      *,*) echo "ERRORE: virgola in ARGS: sbatch --export la usa come separatore"; exit 1 ;;
    esac
    cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TARGET",ARGS="$args" --job-name="slim_$tag" run.slurm)
    if [ "$DRY" = 1 ]; then
      echo "  DRY: $tag  g = $g  shift = $shift  da $(basename "$src") it $it"
      continue
    fi
    mkdir -p "$seed"
    cp "$up" "$seed/A_komega_spinup_reseed.npz"
    cp "$down" "$seed/A_komega_spindown_reseed.npz"
    env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
    echo "  $tag: P = $P, g = $g (shift $shift) da $(basename "$src") it $it"
  done
done
