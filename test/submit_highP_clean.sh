#!/bin/bash
# HPCLEAN (2026-10-01): alta P (>= 0.75) con griglie e opzioni scelte per iterazioni pulite.
#
# Ricetta = PROD (pintmax 12, k_update 8, exact_zero, coda reticolo 100, Sigma 24 k x 97 w, contatto
# analitico oltre k = 8, alpha 0.3) piu':
#   --thouless-q-mode qff-or-zero   pin al piu' alto dei due candidati 2D: qff esatto e la piu' piccola
#                                   Q > 0 (la riga Q = 0 esatta e' numericamente anomala).  Niente
#                                   global-max: inseguiva massimi numerici (1.081 qff a 0.75, 0.05-0.1 qff)
#   --lattice-dw 2.5e-4             reticolo eps/Omega fine: toglie il dente di sega e le righe spurie
#                                   vicino a qff (l'unione in eps non serve: provato, nessun effetto)
#   --angular-feature-nodes 21      21 nodi angolari attorno a kF del partner invece di 7: il polo del
#                                   minoritario e' largo ~1e-4 e 7 nodi lo campionano a caso
#   --sigma-omega-dense-w 0.06      blocco fitto di nodi omega di Sigma in |w| <= 0.06 (un nodo w_base
#   --sigma-omega-dense-stride 2    su due, passo 2e-3): il polarone del minoritario e' largo 0.07 a
#                                   P = 0.75 e 0.02 a 0.90, contro i ~7e-3 fra i 97 nodi uniformi in
#                                   indice (3 nodi su tutta la banda a 0.90).  97 -> 147 nodi (+52%)
#   --sigma-k-kf-offsets 0.03:0.1:0.3:0.6   nodi k di Sigma in piu' a kF (1 -+ d), oltre ai 24: ad alta P
#                                   i vicini di kF_dn erano a 0.5 e 2.2 kF (nessun nodo sulla banda del
#                                   polarone); confina anche la cuspide di kF (niente spike in N(omega))
#   --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only   (aggiunto 2026-10-02) finestre
#                                   dense nella griglia in p del residuo a kF_up e kF_dn (31 -> 73 nodi):
#                                   senza, -ImPi(qff, |Omega|<0.01) e' sovrastimata 2-3x e g_c spostato
#                                   (test/energy_bubble/pwindow_probe.py)
#   --sigma-nk 20                   20 righe k "farthest point" + 8 ancore a kF = 28 righe = 28 core: con 24 + 8
#                                   = 32 righe il pool faceva un secondo giro per 4 righe sole (2026-10-02)
# Costo osservato: 75-104 min per iterazione senza finestre in p e con --omega-chunk 16; run_test ora passa
# --omega-chunk 1000 (tabella del fermione costruita una volta per riga k: Sigma x6-9 piu' veloce).
#
# Sorgenti: per ogni P l'ultima iterazione completa di una run esistente allo STESSO P se c'e' (le cube
# si copiano), altrimenti della piu' vicina SOTTO, con la griglia k rifatta sui kF nuovi
# (test/reseed_regrid.py).  Tag P0pXXX_hpclean, una catena per P, niente scaletta.
#
# Uso (da test/):   DRY=1 ./submit_highP_clean.sh      (stampa e basta)
#                   ./submit_highP_clean.sh
# Variabili:        POLS="0.75 0.80"   TARGET=30   SUFFIX=hpclean
#                   SOURCES="0.75:P0p75_gmax 0.80:P0p80_x75a0p3 ..."  (P:run sorgente)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../seeds_warm

module load python/3.14.3
source ~/venvs/fflo-sc/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLS=(${POLS:-0.75 0.775 0.80 0.825 0.85 0.875 0.90 0.925 0.95 0.975})
TARGET="${TARGET:-30}"
DRY="${DRY:-0}"
SUFFIX="${SUFFIX:-hpclean}"
SOURCES="${SOURCES:-0.75:P0p75_gmax 0.80:P0p80_x75a0p3 0.85:P0p85_x75a0p3 0.90:P0p90_x75a0p3}"
ARGS_BASE="--pintmax 12 --k-update-max 8 --eta-floor exact_zero --lattice-n-tail 100 --sigma-nomega 97 --sigma-nk 20 --high-k-sigma pair-contact --contact-tail --prune-keep 2 --thouless-q-mode qff-or-zero --lattice-dw 2.5e-4 --angular-feature-nodes 21 --sigma-omega-dense-w 0.06 --sigma-omega-dense-stride 2 --sigma-k-kf-offsets 0.03:0.1:0.3:0.6 --p-feature-width 0.08 --p-feature-nodes 21 --p-feature-mode kf_only"

for P in "${POLS[@]}"; do
  tag="P${P/./p}_${SUFFIX}"
  if [ -d "../out/$tag" ]; then
    echo "  $tag: out/$tag esiste gia', salto (la sua catena riprende da sola)"
    continue
  fi
  # sorgente: stesso P se disponibile, altrimenti la piu' vicina sotto
  read -r SP SRUN < <(python3 -c "
import sys
P = float('$P')
src = [(float(a), b) for a, b in (x.split(':') for x in '$SOURCES'.split())]
low = [s for s in src if s[0] <= P + 1e-9]
p, r = max(low) if low else min(src)
print(p, r)")
  read -r it up down < <(python3 -c "import run_test
last, paths = run_test.last_complete_iteration('../out/$SRUN')
assert paths, 'nessuna iterazione completa in ../out/$SRUN'
print(last, paths[0], paths[1])")
  seed="../seeds_warm/$tag"
  args="$ARGS_BASE --up ${seed#../}/A_komega_spinup_reseed.npz --down ${seed#../}/A_komega_spindown_reseed.npz"
  case "$args" in
    *,*) echo "ERRORE: virgola in ARGS ($args): sbatch --export la usa come separatore e taglia il resto"
         echo "       (2026-10-02: --up/--down persi, 10 run partite dal seed di default a P = 0.65)"; exit 1 ;;
  esac
  cmd=(sbatch --export=ALL,TAG="$tag",TARGET="$TARGET",ARGS="$args" --job-name="slim_$tag" run.slurm)
  same=$(python3 -c "print(1 if abs(float('$P') - float('$SP')) < 1e-9 else 0)")
  if [ "$DRY" = 1 ]; then
    echo "  DRY: $tag da $SRUN it $it ($([ "$same" = 1 ] && echo 'stesso P: copia' || echo "regrid da P = $SP")); ${cmd[*]}"
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
  echo "  $tag: P = $P da $SRUN it $it"
done
