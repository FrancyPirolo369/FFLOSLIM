#!/bin/bash
# TEST 1: la cancellazione  Gamma^-1 = g0(Lambda) - Pi(Lambda)  e' esatta?
# Se lo e', ReGamma^-1(qff,0) NON deve dipendere da Lambda.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SIGMA_PN_ANGLE_EXACT_CUT=1 SIGMA_PN_RING_MODE=exact_window SIGMA_PN_Q_GL_N=16
W="${W:-10}"
QPTS=$(python3 -c "
import numpy as np, sys
sys.path.insert(0,'.')
from run_fflo import smooth_multicenter_grid as g
qff=np.sqrt(1.65)-np.sqrt(0.35)
a=g(0.0,2.2,30,[0.0,qff],[0.15,0.06],[12.0,15.0])
b=g(2.2,12.0,12,[3.0],[0.6],[6.0])
print(','.join(f'{v:.15g}' for v in np.unique(np.concatenate([a,b]))))")
for L in "$@"; do
  echo "=== bubble cutoff Lambda = $L ==="
  python3 -m fflo.pairbuild \
    --up seeds/A_komega_spinup_warm.npz --down seeds/A_komega_spindown_warm.npz \
    --out-dir "out/cutoff_L${L}" --workers "$W" --profile turbo \
    --bubble-p-int-max "$L" \
    --q-points "$QPTS" --q-table-max 12.0 \
    --omega-feature-mode zero --omega-feature-half-width 1.35 \
    --omega-feature-nodes 21
done
