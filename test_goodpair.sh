#!/bin/bash
# A/B PULITO: stessa griglia q (quella di produzione), stesso Qmax, stessi seed,
# stesse impostazioni Sigma.  CAMBIA SOLO LA GRIGLIA OMEGA DELLA COPPIA.
# Poi lo stadio density gira su entrambe e si confrontano le densita'.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SIGMA_PN_ANGLE_EXACT_CUT=1 SIGMA_PN_RING_MODE=exact_window SIGMA_PN_Q_GL_N=16
export SIGMA_PN_PINTMAX=6
UP=seeds/A_komega_spinup_warm.npz
DN=seeds/A_komega_spindown_warm.npz
# la griglia q ESATTA di produzione (run_fflo.py: core_n=40, tail riscalata, Qcap=20)
QPTS=$(python3 -c "
import numpy as np, sys; sys.path.insert(0,'.')
from run_fflo import smooth_multicenter_grid as g
qff=np.sqrt(1.65)-np.sqrt(0.35); QMAX=20.0
tail_n=max(19,int(round(19*(QMAX-2.2)/(11.5-2.2))))
a=g(0.0,2.2,40,[0.0,qff],[0.15,0.06],[12.0,15.0]); b=g(2.2,QMAX,tail_n,[3.0],[0.6],[6.0])
print(','.join(f'{v:.15g}' for v in np.unique(np.concatenate([a,b]))))")
echo "nodi q: $(echo $QPTS | tr ',' '\n' | wc -l)"

pair () {  # tag  omega-mode  halfwidth  nodes
  echo "### PAIR $1  (omega-mode=$2 hw=$3 n=$4)"
  /usr/bin/time -f "   tempo: %E" python3 -m fflo.pairbuild \
    --up $UP --down $DN --out-dir "out/gp_$1" --workers 10 --profile turbo \
    --bubble-p-int-max 4 --q-points "$QPTS" --q-table-max 20.0 \
    --omega-feature-mode "$2" --omega-feature-half-width "$3" \
    --omega-feature-nodes "$4"
}
dens () {  # tag
  echo "### DENSITY $1"
  /usr/bin/time -f "   tempo: %E" python3 -m fflo.density \
    --pair-table "out/gp_$1/pair/pair_gamma_table.npz" \
    --seed-up $UP --seed-down $DN \
    --subcritical-delta 1e-3 --sigma-nk 24 --n-theta 32 --sigma-nomega 41 \
    --eta-floor-mode broad --thouless-q-mode qff --high-k-sigma-mode stale \
    --sigma-k-feature-fraction 0 --pole-refind-n-local 61 \
    --mix 0.3 1.0 --out-dir "out/gpd_$1" --iteration-index 1 --workers 10
}
pair base  zero                1.35 21
pair good  zero_and_thresholds 1.35 21
dens base
dens good
echo "### FATTO"
