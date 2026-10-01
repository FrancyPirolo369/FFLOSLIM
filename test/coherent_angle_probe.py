#!/usr/bin/env python3
"""Dove sta il massimo di ReGamma^-1(Q, 0) se la bolla coerente QPxQP e' integrata con la quadratura
angolare della produzione?  Fermioni liberi (eps = k^2, Z = 1, larghezza = gamma_floor 1e-3), P = 0.50.

La regola "kjac" usa Gauss-Chebyshev in u = cos(phi): con pesi uguali pi/n e nodi cos((2j-1)pi/2n) e'
esattamente la regola del punto medio in phi con passo pi/n (48 nodi in produzione -> 0.065 rad).
Vicino al punto di contatto delle due superfici di Fermi l'energia di coppia a |k| = kF_up vale
E(phi) ~ v_dn (a phi^2 - dQ), a = kF_up Q / (2 kF_dn): la fisica della cuspide per Q - qff = dQ sta a
phi < phi_c = sqrt(dQ / a) ~ 0.1-0.17 rad, cioe' su 1-3 nodi.

Calcola S(Q) = (1/pi) int dOmega ImPi_coh(Q, Omega) / Omega su |Omega| < W (la parte di ReGamma^-1 che
contiene la cuspide; il resto e' liscio in Q) con n_kquad = 48 (prod), 160 (gold) e 480, e riporta dove
sta l'estremo rispetto a qff.  In MSCT esatto (A6) la cuspide e' esattamente a qff.

Uso (dalla radice di SLIM):  python3 test/coherent_angle_probe.py [--nquad 48 160 480]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from fflo.analytic_bubble import coherent_impi_kjac  # noqa: E402

INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=float, default=0.50)
    ap.add_argument("--nquad", type=int, nargs="+", default=[48, 160, 480])
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--W", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "coherent_angle_probe.png"))
    a = ap.parse_args(argv)
    kup, kdn = np.sqrt(1 + a.P), np.sqrt(1 - a.P)
    qff = kup - kdn
    k = np.linspace(0.0, 4.0, 4001)
    Eu, Ed = k * k - kup ** 2, k * k - kdn ** 2
    one, gam = np.ones_like(k), np.full_like(k, a.gamma)
    core = np.linspace(-0.02, 0.02, 80, endpoint=False) + 0.00025      # niente Omega = 0 esatto
    tail = np.geomspace(0.02, a.W, 60)[1:]
    om = np.unique(np.concatenate((-tail[::-1], core, tail)))
    x = np.arange(-0.03, 0.0601, 0.0025)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8), layout="constrained")
    for c, nq in zip(COLS, a.nquad):
        t0 = time.time()
        S = []
        for xv in x:
            im = coherent_impi_kjac(qff * (1 + xv), om, k, Eu, one, gam, k, Ed, one, gam,
                                    nk=96, n_kquad=nq, kquad_chunk_size=16, kmax=4.0,
                                    gamma_floor=a.gamma, kf_features=[kup, kdn])
            S.append(np.trapezoid(im / om, om) / np.pi)
        S = np.array(S)
        i = int(np.argmax(np.abs(S - S[0]) * 0 + S)) if np.nanmax(S) - S[x.size // 2] > 0 else int(np.argmin(S))
        ext = "max" if S[i] >= np.max(S) else "min"
        i = int(np.argmax(S)) if ext == "max" else int(np.argmin(S))
        s0 = float(np.interp(0.0, x, S))
        print(f"n_kquad = {nq:4d} (dphi = {np.pi / nq:.4f}): estremo ({ext}) a Q/qff = {1 + x[i]:.4f}, "
              f"S(Q*) - S(qff) = {S[i] - s0:+.3e}   [{time.time() - t0:.0f} s]")
        ax.plot(1 + x, S - s0, "-o", ms=2.5, color=c, label=f"n_kquad = {nq} (Δφ = {np.pi / nq:.3f})")
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set(xlabel="Q / qff", ylabel="S(Q) − S(qff)  (parte di ReΓ⁻¹ con la cuspide)",
           title=f"Bolla coerente QP×QP, fermioni liberi, P = {a.P}: quadratura angolare kjac")
    ax.legend(fontsize=7.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
