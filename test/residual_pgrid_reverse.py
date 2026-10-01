#!/usr/bin/env python3
"""Quanto del massimo oltre qff viene dalla griglia in p della parte residua della bolla?  Ricostruzione a posteriori.

La parte a griglia di ImPi (A*A - A0*A0) si ricalcola sulle righe vicino a qff con test/impi_union_probe.py,
che riproduce impi_table di produzione (ricostruzione di A da Sigma, angolo phipanel 10 + 7 nodi a kF,
reticolo in eps di prod, p lineare su [0, 4] con trapezio): una volta con n_p = 31 (prod) e una con n_p
fine.  Poi, come in qpqp_reverse_probe.py,
    DeltaImGamma^-1 = -(ImPi_fine - ImPi_31),   DeltaReGamma^-1(Q, 0) = (1/pi) int dOmega DeltaImGamma^-1 / Omega
e Gamma^-1_nuovo = Gamma^-1_snap + Delta.  Un passo solo; Delta g = -4 pi DeltaReGamma^-1(qff, 0).

Uso (dalla radice di SLIM):  python3 test/residual_pgrid_reverse.py [RUN] [--np-fine 124] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402
from fflo.qp_cube import build_qp_cube, qp_model_from_cube  # noqa: E402

DELTA = 1e-3
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def omega_grid(w_snap, core=0.05, wmax=2.0, n_mid=120, n_tail=16, wtail=12.0):
    wu = np.unique(w_snap)
    c = wu[(np.abs(wu) <= core) & (np.abs(wu) > 0)]
    mid = np.geomspace(core, wmax, n_mid)[1:]
    tail = np.geomspace(wmax, wtail, n_tail)[1:]
    pos = np.concatenate((mid, tail))
    return np.unique(np.concatenate((-pos[::-1], c, pos)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p50_prod")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--np-fine", type=int, default=124)
    ap.add_argument("--qs", default="0.971,0.989,1.000,1.007,1.025,1.045,1.067")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    snap = sorted(glob.glob(os.path.join(a.base, a.run, "snap", "iter*.npz")))[-1]
    up = sorted(glob.glob(os.path.join(a.base, a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    work = os.path.join(HERE, "out", "impi_probe", "residual_pgrid", a.run)
    os.makedirs(work, exist_ok=True)
    a0u, a0d = os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz")
    for src, dst in ((up, a0u), (dn, a0d)):
        if not os.path.exists(dst):
            build_qp_cube(src, dst, model=qp_model_from_cube(src))
    z = np.load(snap)
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_old = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    targets = [float(x) for x in a.qs.split(",")]
    rows = np.array(sorted({int(np.argmin(np.abs(q / qff - t))) for t in targets}))
    om = omega_grid(w)
    print(f"{a.run}: righe Q/qff {np.round(q[rows] / qff, 3)}, {om.size} Omega in [{om[0]:.1f}, {om[-1]:.1f}]", flush=True)
    ims = {}
    for n_p in (31, a.np_fine):
        t0 = time.time()
        U.setup(up, dn, a0u, a0d, n_p=n_p)
        ims[n_p] = U.run([float(v) for v in q[rows]], om, "lattice", a.workers)
        np.savez(os.path.join(work, f"impi_np{n_p}.npz"), q=q[rows], omega=om, impi=ims[n_p])
        print(f"  n_p = {n_p}: fatto in {time.time() - t0:.0f} s", flush=True)
    dre = np.trapezoid(-(ims[a.np_fine] - ims[31]) / om[None, :], om, axis=1) / np.pi
    for j, iq in enumerate(rows):
        print(f"  Q/qff = {q[iq] / qff:.3f}: DeltaReGamma^-1(Q,0) = {dre[j] / DELTA:+.3f} delta")
    re_new = re_old[rows] + dre
    iqf = int(np.argmin(np.abs(q[rows] - qff)))
    for lab, r in (("snap (residua n_p 31)", re_old[rows]), (f"ricostruita (residua n_p {a.np_fine})", re_new)):
        i = int(np.argmax(r))
        print(f"{lab:32s}: Q*/qff = {q[rows][i] / qff:.3f}, D = {(r[i] - r[iqf]) / DELTA:+.2f} delta")
    print(f"DeltaReGamma^-1(qff) = {dre[iqf] / DELTA:+.2f} delta  ->  Delta g (un passo) = {-4 * np.pi * dre[iqf]:+.4f}")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), layout="constrained")
    ax.plot(q[rows] / qff, (re_old[rows] - re_old[rows][iqf]) / DELTA, "o-", color="#2a78d6", label="snap (residua, 31 nodi p)")
    ax.plot(q[rows] / qff, (re_new - re_new[iqf]) / DELTA, "o-", color="#eb6834", label=f"ricostruita (residua, {a.np_fine} nodi p)")
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q, 0) − ReΓ⁻¹(qff, 0)  [δ]", title=f"{a.run}: griglia in p della parte residua")
    ax.legend(fontsize=7.5)
    out = os.path.join(HERE, "out", "cluster", "plots", f"residual_pgrid_{a.run}.png")
    fig.savefig(out, dpi=115)
    print("scritto", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
