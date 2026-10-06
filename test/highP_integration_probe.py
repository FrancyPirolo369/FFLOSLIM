#!/usr/bin/env python3
"""Alta P: come cambia ReGamma^-1(Q, 0) vicino a Q = 0 e a qff con l'integrazione della bolla.  A un passo.

Sullo stato di una run (cube di next_cubes + tabella grezza della snap della stessa iterazione) si
ricalcola la parte residua a griglia di ImPi (A*A - A0*A0, test/impi_union_probe.py = impi_table di
produzione) sulle righe di Q che decidono la gara fra i canali, con diverse integrazioni:
    MODO,DW,NP  modo = lattice (reticolo in eps di prod) | union (+ nucleo spostato su eps = Omega)
                dw   = passo fine dei reticoli (blocco lineare riscalato a pari larghezza)
                np   = nodi della griglia in p su [0, Lambda = 4]
La differenza da quella della run (--base) passa in ReGamma^-1 con la KK,
    DeltaReGamma^-1(Q, 0) = (1/pi) int dOmega [-(ImPi_var - ImPi_base)] / Omega,
e si legge: righe spurie vicino a qff, rumore vicino a Q = 0, gara ReG^-1(0) - ReG^-1(qff), e
Delta g_c a un passo col pin a due candidati (qff esatto oppure Q = 0).  La parte coerente QPxQP e le
larghezze non cambiano.

Uso (dalla radice di SLIM):
  python3 test/highP_integration_probe.py P0p75_gmax [--base lattice,1e-3,31]
        [--variants union,1e-3,31 lattice,2.5e-4,31 ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import math
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
from residual_pgrid_reverse import omega_grid  # noqa: E402

DELTA = 1e-3
ROWS = [0, 0.014, 0.028, 0.042, 0.057, 0.107, 0.208, 0.398, 0.85, 0.935, 0.968, 0.981, 0.992,
        1.0, 1.004, 1.015, 1.027, 1.041, 1.058, 1.081, 1.122]
VARIANTS = ["lattice,1e-3,31", "union,1e-3,31", "lattice,2.5e-4,31", "union,2.5e-4,31",
            "lattice,1e-3,124", "union,2.5e-4,124"]
COLS = ["#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c0392b", "#8a8a85"]


def parse_variant(s):
    mode, dw, n_p = s.split(",")
    return mode, float(dw), int(n_p)


def compute(variant, rows_q, om, cubes, workers, work):
    mode, dw, n_p = parse_variant(variant)
    tag = "" if len(rows_q) == len(ROWS) else f"_r{len(rows_q)}_{abs(hash(tuple(np.round(rows_q, 6)))) % 10**6}"
    path = os.path.join(work, f"impi_{mode}_dw{dw:g}_np{n_p}{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if np.allclose(z["q"], rows_q) and np.allclose(z["omega"], om):
            return z["impi"], 0.0
    t0 = time.time()
    U.setup(*cubes, n_p=n_p, dw=dw, n_linear=int(round(40 * 1e-3 / dw)))
    impi = U.run([float(v) for v in rows_q], om, mode, workers)
    np.savez(path, q=rows_q, omega=om, impi=impi)
    return impi, time.time() - t0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--base", default="lattice,1e-3,31", help="integrazione con cui la run ha fatto la snap")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cluster", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--rows", default="", help="righe Q/qff separate da virgole (default: ROWS)")
    ap.add_argument("--tag", default="", help="suffisso del nome del grafico")
    a = ap.parse_args(argv)
    up = sorted(glob.glob(os.path.join(a.cluster, a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    it = int(os.path.basename(up).split("iter")[-1][:3])
    z = np.load(os.path.join(a.cluster, a.run, "snap", f"iter{it:03d}.npz"))
    q, w, qff = np.asarray(z["q"], float), np.asarray(z["omega"], float), float(z["qff"])
    re_raw = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    targets = [float(x) for x in a.rows.split(",")] if a.rows else ROWS
    rows = np.array(sorted({int(np.argmin(np.abs(q / qff - t))) for t in targets}))
    rq = q[rows]
    work = os.path.join(HERE, "out", "impi_probe", "highP_integration", a.run)
    os.makedirs(work, exist_ok=True)
    a0u, a0d = os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz")
    for src, dst in ((up, a0u), (dn, a0d)):
        if not os.path.exists(dst):
            build_qp_cube(src, dst, model=qp_model_from_cube(src))
    cubes = (up, dn, a0u, a0d)
    om = omega_grid(w)
    print(f"{a.run} it{it}: {rq.size} righe Q/qff {np.round(rq / qff, 3)}, {om.size} Omega in [{om[0]:.0f}, {om[-1]:.0f}]",
          flush=True)
    base, dt = compute(a.base, rq, om, cubes, a.workers, work)
    print(f"  base {a.base}: {dt:.0f} s", flush=True)
    i0 = int(np.argmin(np.abs(rq)))
    iq = int(np.argmin(np.abs(rq - qff)))
    near0 = rq / qff <= 0.11
    nearq = (rq / qff >= 0.9) & (rq / qff <= 1.13)
    g0 = -0.5 * math.log(2.0)

    def report(lab, r):
        best = max(r[i0], r[iq])
        jn = np.flatnonzero(nearq)[int(np.argmax(r[nearq]))]
        rough = np.std(np.diff(r[near0], 2)) / DELTA
        print(f"  {lab:20s} gara ReG(0)-ReG(qff) = {(r[i0] - r[iq]) / DELTA:+6.2f} delta -> vince "
              f"{'Q=0' if r[i0] > r[iq] else 'qff'} | max vicino a qff a {rq[jn] / qff:.3f} qff, "
              f"{(r[jn] - r[iq]) / DELTA:+5.2f} delta sopra qff | rugosita' a Q~0 {rough:4.2f} delta | "
              f"g_c (2 candidati) = {-4 * math.pi * best + g0:+.4f}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    curves = {}
    report(f"snap ({a.base})", re_raw[rows])
    curves[f"snap ({a.base})"] = re_raw[rows]
    for v in a.variants:
        if v == a.base:
            continue
        impi, dt = compute(v, rq, om, cubes, a.workers, work)
        dre = np.trapezoid(-(impi - base) / om[None, :], om, axis=1) / np.pi
        r = re_raw[rows] + dre
        curves[v] = r
        report(f"{v} [{dt:.0f} s]", r)
    for c, (lab, r) in zip(COLS, curves.items()):
        ref = r[iq]
        for ax, m in ((axs[0], near0), (axs[1], nearq)):
            ax.plot(rq[m] / qff, (r[m] - ref) / DELTA, "o-", color=c, ms=3.5, lw=1.4, label=lab)
    axs[0].set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q,0) − ReΓ⁻¹(qff,0)  [δ]", title=f"{a.run}: canale a Q ≈ 0 (sopra 0 = vince Q≈0)")
    axs[1].set(xlabel="Q / qff", ylabel="ReΓ⁻¹(Q,0) − ReΓ⁻¹(qff,0)  [δ]", title="vicino a qff (in 2D il massimo vero sta a 1)")
    axs[1].axvline(1.0, color="#9a9a96", lw=0.8)
    for ax in axs:
        ax.axhline(0.0, color="#9a9a96", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    dst = os.path.join(HERE, "out", "cluster", "plots", f"highP_integration_{a.run}{a.tag}.png")
    fig.savefig(dst, dpi=115)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
