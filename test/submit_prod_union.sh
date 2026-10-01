#!/bin/bash
# PROD_UNION (2026-09-29): ricetta di PROD + global-max + nodi unione in eps nell'integrale di
# ImPi (pairbuild/impi_table --eps-union-core, run_test --impi-eps-union).  L'unione aggiunge
# per ogni Omega il nucleo del reticolo in eps spostato su eps = Omega (livello di Fermi del
# minoritario) e calcola l'integrando esattamente sui nodi: toglie il dente di sega sui nodi
# del reticolo e il glitch della riga Q = 0 (validato in locale, test/impi_union_probe.py).
# Il massimo globale ora e' cercato solo in Q <= 2 qff (fflo/density.py), per tutte le run.
#
# Ogni P riparte dall'ultima iterazione completa di una run esistente allo STESSO P (le cube
# si copiano e basta).  Sorgente di default: P0pXX_x75a0p3 (le catene ad alta P di
# submit_highP_x.sh); SRC_SUFFIX=prod per ripartire dalle PROD.  Tag P0pXX_prod_union,
# nessuna scaletta.
#
# Uso (da test/):   ./submit_prod_union.sh          DRY=1 ./submit_prod_union.sh = solo stampa
# Variabili:        POLS="0.80 0.85 0.90"  SRC_SUFFIX=x75a0p3  UNION=0.12  TARGET=30
#                   DW=2.5e-4 -> reticolo eps/Omega fine (risolve il polo del minoritario a kF ad
#                   alta P, toglie le righe Q spurie vicino a qff); tag P0pXX_prod_union_fine
#                   QMODE=qff -> pin a qff come PROD invece di global-max; tag con suffisso _qff.
#                   In 2D il massimo vero sta a qff = kF_up - kF_dn: con QMODE=qff la sola differenza
#                   da PROD e' il reticolo, e il dislivello D = ReG^-1(Q*) - ReG^-1(qff) nelle snap
#                   misura quanto il reticolo sposta il massimo (test/ablation_report.py, 2026-09-30)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
# sul nodo di login OpenBLAS con i thread di default non riesce ad allocare i buffer
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLS=(${POLS:-0.80 0.85 0.90})
SRC_SUFFIX="${SRC_SUFFIX:-x75a0p3}"
UNION="${UNION:-0.12}"
DW="${DW:-}"          # passo fine dei reticoli Omega/eps; vuoto = 1e-3.  2.5e-4 -> tag _prod_union_fine
TARGET="${TARGET:-30}"
QMODE="${QMODE:-global-max}"   # global-max (alte P, gara con Q~0) oppure qff (come PROD)
DRY="${DRY:-0}"
ARGS_BASE="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --sigma-nomega 97 --high-k-sigma pair-contact --contact-tail --thouless-q-mode $QMODE --impi-eps-union $UNION --prune-keep 2"

SUFFIX=prod_union
if [ -n "$DW" ]; then
  ARGS_BASE="$ARGS_BASE --lattice-dw $DW"
  SUFFIX=prod_union_fine
fi
if [ "$QMODE" != global-max ]; then
  SUFFIX="${SUFFIX}_${QMODE//-/}"
fi

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
  cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TARGET",ARGS="$args" --job-name="slim_$tag" run.slurm)
  if [ "$DRY" = 1 ]; then
    echo "  DRY: cp $up $down -> $seed/ ; ${cmd[*]}"
  else
    mkdir -p "$seed"
    cp "$up" "$seed/A_komega_spinup_reseed.npz"
    cp "$down" "$seed/A_komega_spindown_reseed.npz"
    env -u LADDER -u LADDER_ARGS -u LADDER_SUFFIX "${cmd[@]}"
    echo "  $tag: P = $P, union = $UNION, dw = ${DW:-1e-3}, q-mode = $QMODE, da $(basename "$src") iterazione $it"
  fi
done
