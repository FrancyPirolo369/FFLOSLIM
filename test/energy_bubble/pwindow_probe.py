#!/usr/bin/env python3
"""Griglie in p STATICHE per la parte residua della bolla, contro il riferimento adattivo, sulla riga qff.

Per ogni run gia' passata da eb_qff_scan.py (out/energy_bubble/qff_scan/RUN/qff_row.npz: ImPi vecchio e
adattivo sulla stessa griglia in Omega) si ricalcola il residuo A*A - A0*A0 come impi_table (trapezio in
p, reticolo dw 1e-3, phipanel 10 + 7) con griglie in p non uniformi:
    31 uniformi su [0, 4] + una finestra attorno a kF_up (che non dipende da Q: alla tangenza qff + kF_dn
    = kF_up, e li' il residuo ha una buca stretta)
e si confronta DeltaReGamma^-1(qff, 0) = (1/pi) int dOmega [-(ImPi_griglia - ImPi_adattivo)] / Omega:
quanto manca ancora al riferimento, in delta e in g_c a un passo.

Uso (dalla radice di SLIM):  python3 test/energy_bubble/pwindow_probe.py [RUN ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402

RUNS = ["P0p10_prod", "P0p20_prod", "P0p30_prod", "P0p40_prod", "P0p50_prod", "P0p60_prod",
        "P0p65_prod", "P0p70_gmax"]
DELTA = 1e-3


def graded(c, dmin, dmax, n_side):
    """c e n_side nodi per lato a distanza geometrica da dmin a dmax."""
    d = np.geomspace(dmin, dmax, n_side)
    return np.concatenate([c - d[::-1], [c], c + d])


def grids(kfu, kfd):
    uni = np.linspace(0.0, 4.0, 31)
    clip = lambda g: np.unique(g[(g >= 0.0) & (g <= 4.0)])
    return {
        "prod: 31 uniformi": uni,
        "31 + 41 unif. in kFu+-0.15": clip(np.concatenate([uni, np.linspace(kfu - 0.15, kfu + 0.15, 41)])),
        "31 + graduata kFu (12/lato, 1e-3..0.3)": clip(np.concatenate([uni, graded(kfu, 1e-3, 0.3, 12)])),
        "31 + graduata kFu (20/lato, 5e-4..0.3)": clip(np.concatenate([uni, graded(kfu, 5e-4, 0.3, 20)])),
        "31 + grad. kFu (20/lato) + kFd (10/lato)": clip(np.concatenate([uni, graded(kfu, 5e-4, 0.3, 20),
                                                                         graded(kfd, 1e-3, 0.2, 10)])),
        "124 uniformi": np.linspace(0.0, 4.0, 124),
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=RUNS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    table = {}
    for run in a.runs:
        work = os.path.join(HERE, "out", "energy_bubble", "qff_scan", run)
        r = np.load(os.path.join(work, "qff_row.npz"))
        om, coh, new = r["omega"], r["coherent"], r["new"]
        up = sorted(glob.glob(os.path.join(HERE, "out", "cluster", run, "iter*", "next_cubes",
                                           "A_komega_spinup_iter*.npz")))[-1]
        dn = up.replace("spinup", "spindown")
        qff = U.setup(up, dn, os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz"),
                      n_p=31, dw=1e-3, n_linear=40)
        kfu, kfd = U.G["kf"]
        print(f"== {run}: qff = {qff:.4f}, kF_up = {kfu:.4f}, kF_dn = {kfd:.4f}", flush=True)
        for lab, g in grids(kfu, kfd).items():
            U.G["p"] = g
            t0 = time.time()
            res = U.run([qff], om, "lattice", a.workers)[0]
            d = float(np.trapezoid(-((res + coh) - new) / om, om) / np.pi)
            small = np.abs(om) < 0.01
            ds = float(np.trapezoid(np.where(small, -((res + coh) - new) / om, 0.0), om) / np.pi)
            table.setdefault(lab, {})[run] = -4 * math.pi * d
            print(f"   {lab:42s} {g.size:4d} nodi: manca al riferimento {d / DELTA:+6.2f} delta "
                  f"(di cui |Om|<0.01: {ds / DELTA:+6.2f}) -> errore su g_c {-4 * math.pi * d:+.4f}  "
                  f"[{time.time() - t0:.1f} s]", flush=True)
    print("\nerrore residuo su g_c (un passo) per griglia e P:")
    print(f"{'griglia':42s} " + " ".join(f"{r[1:5].replace('p', '.'):>7s}" for r in a.runs))
    for lab, row in table.items():
        print(f"{lab:42s} " + " ".join(f"{row[r]:+7.4f}" for r in a.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
