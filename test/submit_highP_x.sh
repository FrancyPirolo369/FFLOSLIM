#!/bin/bash
# Prova 2026-09-28: P alte (0.80 0.85 0.90) ripartendo dall'ultima iterazione completa di
# out/P0p75_gmax (griglia k rifatta sui kF nuovi, test/reseed_regrid.py), ricetta di PROD +
# global-max come la scaletta HI di submit.sh, a due mixing: alpha 0.3 (quello di sempre)
# e 0.6 (sperimentale).  Una catena per (P, alpha), niente scaletta.
#
# Cosa aspettarsi: il reseed si porta dietro la componente di coppia del minoritario
# (~0.12 fuori kF a P = 0.75), piu' grande di tutto il budget (1-P)/2 a P >= 0.80.  Gap_dn
# di partenza previsto ~+40% (0.80), ~+80% (0.85), ~+160% (0.90).  Con alpha 0.6 lo shock
# si smaltisce circa il doppio piu' in fretta se il loop contrae, ma allo stesso modo
# accelera un'eventuale fuga (a 0.75 C e' salito per 4 iterazioni prima di girare).
#
# I tag sono nuovi (P0pXX_x75a0pY): non collidono con le scalette HI in corsa, che a
# it 30 creano P0p80_gmax, P0p85_gmax, P0p90_gmax.  Non serve cancellare niente.
# Per il diagramma di fase:  plot_phase_diagram.py --families prodg,gmax,prod,x75a0p3
#
# Uso (da test/):   ./submit_highP_x.sh           DRY=1 ./submit_highP_x.sh = solo stampa
# Variabili:        POLS="0.80 0.85 0.90"  ALPHAS="0.3 0.6"  SRC=../out/P0p75_gmax  TARGET=30
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
# sul nodo di login OpenBLAS con i thread di default non riesce ad allocare i buffer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SRC="${SRC:-../out/P0p75_gmax}"
POLS=(${POLS:-0.80 0.85 0.90})
ALPHAS=(${ALPHAS:-0.3 0.6})
TARGET="${TARGET:-30}"
DRY="${DRY:-0}"
HI_ARGS="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --sigma-nomega 97 --high-k-sigma pair-contact --contact-tail --thouless-q-mode global-max --prune-keep 2"

read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('$SRC')
assert paths, 'nessuna iterazione completa in $SRC'
print(last, paths[0], paths[1])")
src_tag=$(basename "$SRC")          # P0p75_gmax
short="x${src_tag:3:2}"             # x75
echo "== sorgente: $SRC iterazione $it"

for P in "${POLS[@]}"; do
  seed="../seeds_warm/P${P/./p}_${short}"
  # un seed per P, condiviso dagli alpha; si rifa' solo se manca (cancella la cartella
  # per ripartire da un'iterazione piu' recente della sorgente)
  if [ ! -f "$seed/A_komega_spinup_reseed.npz" ]; then
    if [ "$DRY" = 1 ]; then
      echo "  DRY: python3 reseed_regrid.py --up $up --down $down --P $P --out-dir $seed"
    else
      python3 reseed_regrid.py --up "$up" --down "$down" --P "$P" --out-dir "$seed"
    fi
  fi
  for a in "${ALPHAS[@]}"; do
    tag="P${P/./p}_${short}a${a/./p}"
    if [ -d "../out/$tag" ]; then
      echo "  $tag: out/$tag esiste gia', salto (la sua catena riprende da sola)"
      continue
    fi
    args="$HI_ARGS --alpha $a --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
    cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TARGET",ARGS="$args" --job-name="slim_$tag" run.slurm)
    if [ "$DRY" = 1 ]; then
      echo "  DRY: ${cmd[*]}"
    else
      # niente LADDER: queste catene si fermano a TARGET
      env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
      echo "  $tag: P = $P, alpha = $a, da $src_tag it $it"
    fi
  done
done
