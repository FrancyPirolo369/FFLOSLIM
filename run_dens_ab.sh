#!/bin/bash
# density su due tabelle di coppia con la STESSA griglia q (42 nodi, Qmax=12)
# e la sola griglia omega diversa.  Qmax=12, pintmax=6  =>  k_update_max = 6.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SIGMA_PN_ANGLE_EXACT_CUT=1 SIGMA_PN_RING_MODE=exact_window SIGMA_PN_Q_GL_N=16
export SIGMA_PN_PINTMAX=6
UP=seeds/A_komega_spinup_warm.npz
DN=seeds/A_komega_spindown_warm.npz
for tag in "$@"; do
  echo "### DENSITY  $tag   ($(python3 -c "
import numpy as np;z=np.load('out/om_$tag/pair/pair_gamma_table.npz')
print(f\"nQ={z['q'].size} nW={z['omega'].size}\")"))"
  /usr/bin/time -f "   tempo: %E" python3 -m fflo.density \
    --pair-table "out/om_$tag/pair/pair_gamma_table.npz" \
    --seed-up $UP --seed-down $DN \
    --subcritical-delta 1e-3 --sigma-nk 24 --n-theta 32 --sigma-nomega 41 \
    --eta-floor-mode broad --thouless-q-mode qff --high-k-sigma-mode stale \
    --sigma-k-feature-fraction 0 --pole-refind-n-local 61 \
    --mix 0.3 1.0 --out-dir "out/dens_$tag" --iteration-index 1 --workers 10
  echo "### $tag FATTO"
done
