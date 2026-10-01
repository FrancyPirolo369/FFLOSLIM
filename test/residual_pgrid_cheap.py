#!/usr/bin/env python3
"""Griglia in p economica per la parte residua della bolla: 31 nodi uniformi + finestre fitte attorno a kF.

Riusa test/residual_pgrid_reverse.py (stesse righe Q vicino a qff, stessa ricetta di prod): confronta, per
ogni griglia, DeltaReGamma^-1(Q, 0) rispetto a quella di prod (31 uniformi) con il riferimento a 124 nodi
(gia' calcolato e salvato in out/impi_probe/residual_pgrid/RUN/impi_np{31,124}.npz).  Vicino a qff lo spigolo
del partner, p = kF_dn + Q, cade su kF_up, quindi finestre attorno a kF_up e kF_dn coprono anche il guscio.

Uso (dalla radice di SLIM):  python3 test/residual_pgrid_cheap.py [RUN] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402

DELTA = 1e-3


def grid(kfs, half, n_local, n_uni=31, pmax=4.0):
    g = [np.linspace(0.0, pmax, n_uni)]
    for kf in kfs:
        g.append(np.linspace(kf - half, kf + half, n_local))
    return np.unique(np.concatenate(g))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p50_prod")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args(argv)
    work = os.path.join(HERE, "out", "impi_probe", "residual_pgrid", a.run)
    r31, r124 = np.load(os.path.join(work, "impi_np31.npz")), np.load(os.path.join(work, "impi_np124.npz"))
    qrows, om = r31["q"], r31["omega"]
    snap = sorted(glob.glob(os.path.join(a.base, a.run, "snap", "iter*.npz")))[-1]
    up = sorted(glob.glob(os.path.join(a.base, a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    z = np.load(snap)
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_old = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    rows = np.array([int(np.argmin(np.abs(q - v))) for v in qrows])
    iqf = int(np.argmin(np.abs(q[rows] - qff)))

    def report(lab, im, npts):
        dre = np.trapezoid(-(im - r31["impi"]) / om[None, :], om, axis=1) / np.pi
        r = re_old[rows] + dre
        i = int(np.argmax(r))
        ref = np.trapezoid(-(r124["impi"] - r31["impi"]) / om[None, :], om, axis=1) / np.pi
        rms = np.sqrt(np.mean(((dre - dre[iqf]) - (ref - ref[iqf])) ** 2)) / DELTA
        print(f"{lab:30s} {npts:4d} nodi: Q*/qff = {q[rows][i] / qff:.3f}, D = {(r[i] - r[iqf]) / DELTA:+.2f} delta, "
              f"forma vs 124 uniformi: RMS {rms:.2f} delta", flush=True)

    report("prod (31 uniformi)", r31["impi"], 31)
    report("riferimento (124 uniformi)", r124["impi"], 124)
    U.setup(up, dn, os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz"), n_p=31)
    kfs = U.G["kf"]
    ku, kd = kfs
    shells = [abs(kd - qff), kd, ku, ku + qff]          # gusci del partner a Q = qff (kd + qff = ku, ku - qff = kd)
    variants = [("62 uniformi", np.linspace(0.0, 4.0, 62)),
                ("31 + finestre sui 4 gusci +-0.06 x15", grid(shells, 0.06, 15)),
                ("31 + gusci +-0.06 x15 + kF +-0.02 x11", np.unique(np.concatenate((grid(shells, 0.06, 15), grid(kfs, 0.02, 11)))))]
    for lab, pg in variants[a.first:]:
        U.G["p"] = pg
        t0 = time.time()
        im = U.run([float(v) for v in qrows], om, "lattice", a.workers)
        report(lab, im, pg.size)
        print(f"    [{time.time() - t0:.0f} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
