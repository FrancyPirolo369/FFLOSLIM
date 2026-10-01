#!/bin/bash
# TEST 2: la griglia omega risolve lo stato legato a due corpi (omega = -3.0)?
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SIGMA_PN_ANGLE_EXACT_CUT=1 SIGMA_PN_RING_MODE=exact_window SIGMA_PN_Q_GL_N=16
QPTS=$(python3 -c "
import numpy as np, sys; sys.path.insert(0,'.')
from run_fflo import smooth_multicenter_grid as g
qff=np.sqrt(1.65)-np.sqrt(0.35)
a=g(0.0,2.2,30,[0.0,qff],[0.15,0.06],[12.0,15.0]); b=g(2.2,12.0,12,[3.0],[0.6],[6.0])
print(','.join(f'{v:.15g}' for v in np.unique(np.concatenate([a,b]))))")
run () {  # tag  mode  halfwidth  nodes
  echo "=== $1 : mode=$2 hw=$3 nodes=$4 ==="
  python3 -m fflo.pairbuild \
    --up seeds/A_komega_spinup_warm.npz --down seeds/A_komega_spindown_warm.npz \
    --out-dir "out/om_$1" --workers 10 --profile turbo --bubble-p-int-max 4 \
    --q-points "$QPTS" --q-table-max 12.0 \
    --omega-feature-mode "$2" --omega-feature-half-width "$3" \
    --omega-feature-nodes "$4"
}
run zero        zero                1.35 21
run thresh      zero_and_thresholds 1.35 21
run paper       paper_thresholds    1.35 21
run thresh_wide zero_and_thresholds 2.50 41
