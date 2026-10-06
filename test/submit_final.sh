#!/bin/bash
# FINAL (2026-10-02): una sola ricetta, con tutte le correzioni trovate, per tutta la linea critica.
#
# Base = PROD (pintmax 12, k_update 8, floor eta exact_zero, coda del reticolo 100, contatto analitico
# oltre k = 8, alpha 0.3) piu':
#   bolla
#     --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only   finestre in p a kF_up/kF_dn
#        (31 -> 73 nodi): il residuo A*A - A0*A0 ha una buca stretta a kF_up, senza -ImPi(qff,|Omega|<0.01)
#        e' sovrastimata 2-3x e g_c e' spostato di 0.02-0.08 a un passo (test/energy_bubble/*)
#     --lattice-dw 2.5e-4                       reticolo eps/Omega fine: niente dente di sega vicino a qff
#     --angular-feature-nodes 21                polo del partner (largo ~1e-4 ad alta P) campionato bene
#     (default di run_test) --impi-eps-window   residuo solo nella finestra T = 0 [0, Omega]: identico, x2.5
#   Sigma
#     --sigma-nomega 193 --sigma-omega-dense-w 0.06 --sigma-omega-dense-stride 2   nodi omega fitti vicino a
#        0 (pavimento di ImSigma(kF, 0), banda stretta del polarone ad alta P)
#     --sigma-nk 20 --sigma-k-kf-offsets 0.03:0.1:0.3:0.6   righe k a kF(1 -+ d): niente spike in N(omega);
#        20 + 8 = 28 righe = 28 core (un solo giro del pool)
#     (default di run_test) --omega-chunk 1000  tabella del fermione una volta per riga k: Sigma x6-9
#   pin
#     --thouless-q-mode qff-or-zero             qff esatto o la piu' piccola Q > 0, il piu' alto
#
# Sorgenti: per ogni P l'ultima iterazione completa della run elencata in SOURCES allo stesso P (copia),
# altrimenti della piu' vicina sotto con la griglia k rifatta (test/reseed_regrid.py).  Tag P0pXXX_final.
# TARGET: TARGET_LO (default 15) per P < 0.70 (partono da stati convergiti), TARGET_HI (default 30) sopra.
#
# Uso (da test/):   DRY=1 ./submit_final.sh       POLS="0.30 0.50" ./submit_final.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLS=(${POLS:-0.10 0.20 0.30 0.40 0.50 0.60 0.65 0.70 0.75 0.80 0.85 0.90})
TARGET_LO="${TARGET_LO:-15}"
TARGET_HI="${TARGET_HI:-30}"
SUFFIX="${SUFFIX:-final}"
DRY="${DRY:-0}"
SOURCES="${SOURCES:-0.10:P0p10_prod 0.20:P0p20_prod 0.30:P0p30_prod 0.40:P0p40_prod 0.50:P0p50_prod 0.60:P0p60_prod 0.65:P0p65_prod 0.70:P0p70_gmax 0.75:P0p75_gmax 0.80:P0p80_x75a0p3 0.85:P0p85_x75a0p3 0.90:P0p90_x75a0p3}"
ARGS_BASE="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --high-k-sigma pair-contact --contact-tail --prune-keep 2 --thouless-q-mode qff-or-zero --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only --lattice-dw 2.5e-4 --angular-feature-nodes 21 --sigma-nomega 193 --sigma-omega-dense-w 0.06 --sigma-omega-dense-stride 2 --sigma-nk 20 --sigma-k-kf-offsets 0.03:0.1:0.3:0.6"

for P in "${POLS[@]}"; do
  tag="P${P/./p}_${SUFFIX}"
  if [ -d "../out/$tag" ]; then
    echo "  $tag: out/$tag esiste gia', salto (la sua catena riprende da sola)"
    continue
  fi
  read -r SP SRUN TGT < <(python3 -c "
P = float('$P')
src = [(float(a), b) for a, b in (x.split(':') for x in '$SOURCES'.split())]
low = [s for s in src if s[0] <= P + 1e-9]
p, r = max(low) if low else min(src)
print(p, r, $TARGET_LO if P < 0.70 - 1e-9 else $TARGET_HI)")
  read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('../out/$SRUN')
assert paths, 'nessuna iterazione completa in ../out/$SRUN'
print(last, paths[0], paths[1])")
  seed="../seeds_warm/$tag"
  args="$ARGS_BASE --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
  case "$args" in
    *,*) echo "ERRORE: virgola in ARGS: sbatch --export la usa come separatore"; exit 1 ;;
  esac
  cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TGT",ARGS="$args" --job-name="slim_$tag" run.slurm)
  same=$(python3 -c "print(1 if abs(float('$P') - float('$SP')) < 1e-9 else 0)")
  if [ "$DRY" = 1 ]; then
    echo "  DRY: $tag (TARGET $TGT) da $SRUN it $it ($([ "$same" = 1 ] && echo 'stesso P: copia' || echo "regrid da P = $SP"))"
    continue
  fi
  mkdir -p "$seed"
  if [ "$same" = 1 ]; then
    cp "$up" "$seed/A_komega_spinup_reseed.npz"
    cp "$down" "$seed/A_komega_spindown_reseed.npz"
  else
    python3 reseed_regrid.py --up "$up" --down "$down" --P "$P" --out-dir "$seed"
  fi
  env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
  echo "  $tag: P = $P (TARGET $TGT) da $SRUN it $it"
done
