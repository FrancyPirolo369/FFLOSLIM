#!/usr/bin/env python3
"""Prova veloce: griglie STATICHE infittite dove serve nella parte residua di ImPi (impi_table di produzione).

Come test/highP_integration_probe.py (stesse righe, stessa KK della differenza, stessa base = la ricetta
della snap), ma con griglie in p non uniformi: 31 nodi uniformi su [0, 4] piu' finestre fitte attorno a
kF_up (dove il residuo ha la buca) e a kF_dn, e con piu' nodi angolari attorno a kF del partner.

Uso (dalla radice di SLIM):  python3 test/static_grid_probe.py [RUN] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402
from residual_pgrid_reverse import omega_grid  # noqa: E402

DELTA = 1e-3
ROWS = [0, 0.014, 0.028, 0.042, 0.057, 0.107, 0.992, 1.0, 1.004, 1.015, 1.027, 1.041]


def pgrid(kfu, kfd, n_up, w_up, n_dn, w_dn):
    g = [np.linspace(0.0, 4.0, 31), np.linspace(kfu - w_up, kfu + w_up, n_up)]
    if n_dn:
        g.append(np.linspace(max(0.0, kfd - w_dn), kfd + w_dn, n_dn))
    return np.unique(np.concatenate(g))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p75_gmax")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    base = os.path.join(HERE, "out", "cluster", a.run)
    up = sorted(glob.glob(os.path.join(base, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    it = int(os.path.basename(up).split("iter")[-1][:3])
    z = np.load(os.path.join(base, "snap", f"iter{it:03d}.npz"))
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_snap = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    rows = np.array(sorted({int(np.argmin(np.abs(q / qff - t))) for t in ROWS}))
    rq = q[rows]
    om = omega_grid(w)
    work = os.path.join(HERE, "out", "impi_probe", "static_grid", a.run)
    os.makedirs(work, exist_ok=True)
    a0u = os.path.join(HERE, "out", "impi_probe", "highP_integration", a.run, "A0_up.npz")
    a0d = a0u.replace("A0_up", "A0_dn")
    iq = int(np.argmin(np.abs(rq - qff)))
    i0 = int(np.argmin(np.abs(rq)))

    def compute(tag, dw, n_kquad, kfeat, grid_fn):
        path = os.path.join(work, f"{tag}.npz")
        if os.path.exists(path):
            return np.load(path)["impi"], np.load(path)["p"].size, 0.0
        t0 = time.time()
        U.setup(up, dn, a0u, a0d, n_p=31, dw=dw, n_linear=int(round(40 * 1e-3 / dw)), n_kquad=n_kquad, kquad_feat=kfeat)
        if grid_fn is not None:
            U.G["p"] = grid_fn(*U.G["kf"])
        impi = U.run([float(v) for v in rq], om, "lattice", a.workers)
        np.savez(path, q=rq, omega=om, impi=impi, p=U.G["p"])
        return impi, U.G["p"].size, time.time() - t0

    variants = [
        ("base_snap", 1e-3, 10, 7, None),                                    # la ricetta della snap
        ("dw_fine", 2.5e-4, 10, 7, None),
        ("dw_fine_finestre", 2.5e-4, 10, 7, lambda u, d: pgrid(u, d, 41, 0.15, 21, 0.10)),
        ("dw_fine_finestre_ang21", 2.5e-4, 10, 21, lambda u, d: pgrid(u, d, 41, 0.15, 21, 0.10)),
        ("dw_fine_finestre2x_ang21", 2.5e-4, 10, 21, lambda u, d: pgrid(u, d, 81, 0.15, 41, 0.10)),
    ]
    base_impi = None
    print(f"{a.run} it{it}: righe Q/qff {np.round(rq / qff, 3)}")
    print(f"{'variante':28s} {'nodi p':>6s} {'tempo':>6s} | " + " ".join(f"{x / qff:6.3f}" for x in rq)
          + " | dReG(qff) d   g_c(qff)  gara")
    for tag, dw, nkq, kf, fn in variants:
        impi, n_p, dt = compute(tag, dw, nkq, kf, fn)
        if base_impi is None:
            base_impi = impi
        r = re_snap[rows] + np.trapezoid(-(impi - base_impi) / om[None, :], om, axis=1) / np.pi
        g = -4 * math.pi * r[iq] - 0.5 * math.log(2)
        print(f"{tag:28s} {n_p:6d} {dt:5.0f}s | " + " ".join(f"{(x - r[iq]) / DELTA:+6.2f}" for x in r)
              + f" | {(r[iq] - re_snap[rows][iq]) / DELTA:+6.2f}  {g:+.4f}  {(r[i0] - r[iq]) / DELTA:+5.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
