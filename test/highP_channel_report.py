#!/usr/bin/env python3
"""Gara fra i canali ad alta P: ReGamma^-1(Q, 0) a Q = 0 contro Q = qff, dalle snap.

In 2D i candidati fisici sono due: il ramo FFLO a qff = kF_up - kF_dn ESATTO e il canale a
Q = 0 (pBCS).  Per ogni run e iterazione, dalla tabella GREZZA della snap e dallo shift che
density ha davvero applicato (shift_used, al Q scelto da --thouless-q-mode):
  pin(Q) = ReGamma^-1(Q, 0) - shift_used - delta     (= -delta al Q del pin)
  pin(0), pin(qff) in unita' di delta: > 0 = quel canale e' OLTRE la criticita'
  g_c applicato e g_c "a due candidati" = -4 pi max(raw(0), raw(qff)) - ln2/2
Il secondo e' a un passo (stessa tabella, nessuna autoconsistenza): dice quale canale
vincerebbe e di quanto si sposterebbe g_c, non il punto fisso.

Uso (dalla radice di SLIM):  python3 test/highP_channel_report.py [RUN ...] [--all-iters]
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "out", "cluster")
DELTA = 1.0e-3
G0 = -0.5 * math.log(2.0)


def gaps(run):
    try:
        txt = open(os.path.join(BASE, run, "loop.log"), errors="replace").read()
    except OSError:
        return {}
    return {int(i): (float(u), float(d)) for i, u, d in
            re.findall(r"iter (\d+) GAP: up=([-+.\d]+)% down=([-+.\d]+)%", txt)}


def row(run, path, gp):
    z = np.load(path)
    q, w, qff = z["q"].astype(float), z["omega"].astype(float), float(z["qff"])
    re0 = z["ReInvGamma"].astype(float)[:, int(np.argmin(np.abs(w)))]
    sh = float(z["shift_used"])
    it = int(os.path.basename(path)[4:7])
    i0 = int(np.argmin(np.abs(q)))
    iq = int(np.argmin(np.abs(q - qff)))
    win = q <= 2 * qff
    im = int(np.argmax(np.where(win, re0, -np.inf)))
    low = q <= 0.5 * qff
    il = int(np.argmax(np.where(low, re0, -np.inf)))
    pin = lambda r: (r - sh - DELTA) / DELTA
    g_app = -4 * math.pi * sh + G0
    best = max(re0[i0], re0[iq])
    g_two = -4 * math.pi * best + G0
    u, d = gp.get(it, (float("nan"), float("nan")))
    # le snap vecchie non hanno q_selected: il Q del pin e' dove la riga grezza vale lo shift
    qpin = float(z["q_selected"]) if "q_selected" in z.files else float(q[int(np.argmin(np.abs(re0 - sh)))])
    return (f"{run:24s} it{it:02d}  Qpin/qff={qpin / qff:5.3f}  "
            f"pin(0)={pin(re0[i0]):+7.1f}  pin(qff)={pin(re0[iq]):+7.1f}  "
            f"max a {q[im] / qff:5.3f} qff ({pin(re0[im]):+6.1f})  max basso a {q[il] / qff:5.3f}  "
            f"C={float(z['pair_contact']):.3f}  g_app={g_app:+.3f}  g_2cand={g_two:+.3f} "
            f"[{'Q=0' if re0[i0] > re0[iq] else 'qff'}]  gap_dn={d:+.1f}%")


def main(argv):
    all_iters = "--all-iters" in argv
    runs = [a for a in argv if not a.startswith("--")]
    if not runs:
        runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(BASE, "P0p[789]*"))
                      if glob.glob(os.path.join(d, "snap", "iter*.npz")))
    print("pin(Q) = ReGamma^-1(Q,0) - shift applicato - delta, in delta: > 0 = oltre la criticita'")
    for run in runs:
        snaps = sorted(glob.glob(os.path.join(BASE, run, "snap", "iter*.npz")))
        if not snaps:
            continue
        gp = gaps(run)
        for p in (snaps if all_iters else snaps[-1:]):
            print(row(run, p, gp))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
