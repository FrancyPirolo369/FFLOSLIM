#!/usr/bin/env python3
"""Doppia buca nella DOS da fluttuazioni FFLO: dipende solo dalla geometria delle superfici di Fermi.

Modello minimo (pseudogap da fluttuazioni statiche su tutto l'anello |Q| = Q0, gas libero, eps = k^2):
    Sigma_dn(k, w) = D^2 < 1 / (w + xi_up(Q - k) + i g) >_{direzioni di Q}
    Sigma_up(p, w) = D^2 < 1 / (w + xi_dn(Q - p) + i g) >
media analitica: 2D  1 / sqrt((z + a)^2 - b^2),  3D  ln((z + a + b) / (z + a - b)) / (2 b),
con a = Q0^2 + k^2 - mu_partner, b = 2 Q0 k.  N(w) = int k^(d-1) dk A(k, w), normalizzata a D = 0.

Previsione: la superficie di accoppiamento xi_dn(k) + xi_up(Q - k) = 0 tocca la superficie a energia
costante del minoritario a  w = -wc  e del maggioritario a  w = +wc, con
    wc = dQ * v_up v_dn / (v_up + v_dn),   dQ = Q0 - (kF_up - kF_dn).
Sfere secanti (3D, dQ > 0): buche a -wc (dn) e +wc (up).  Tangenti (2D, dQ = 0): entrambe a w = 0.

Uso (dalla radice di SLIM):  python3 test/dip_geometry_model.py [--out out/cluster/plots/dip_geometry_model.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
C_DN, C_UP = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def ring_avg(z, a, b, dim):
    if dim == 2:
        return 1.0 / (np.sqrt(z + a - b) * np.sqrt(z + a + b))
    return np.log((z + a + b) / (z + a - b)) / (2 * b)


def dos(dim, kf, kf_partner, q0, delta, gamma, w):
    k = np.linspace(1e-4, 2.2 * max(kf, kf_partner), 24000)[None, :]
    z = w[:, None] + 1j * gamma
    sig = delta ** 2 * ring_avg(z, q0 ** 2 + k ** 2 - kf_partner ** 2, 2 * q0 * k, dim)
    a = -np.imag(1.0 / (z - (k ** 2 - kf ** 2) - sig)) / np.pi
    return np.trapezoid(a * k ** (dim - 1), k[0], axis=1)


def case(dim, kup, kdn, dq, delta, gamma, w):
    q0 = kup - kdn + dq
    ndn = dos(dim, kdn, kup, q0, delta, gamma, w) / dos(dim, kdn, kup, q0, 0.0, gamma, w)
    nup = dos(dim, kup, kdn, q0, delta, gamma, w) / dos(dim, kup, kdn, q0, 0.0, gamma, w)
    vu, vd = 2 * kup, 2 * kdn
    return ndn, nup, dq * vu * vd / (vu + vd)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta", type=float, default=0.012)
    ap.add_argument("--gamma", type=float, default=0.002)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "dip_geometry_model.png"))
    a = ap.parse_args(argv)
    w = np.linspace(-0.08, 0.08, 321)
    p3 = 0.434                                   # tesi Gori: punto critico FFLO a unitarieta', Q0 = 1.076 (kF_up - kF_dn)
    ku3, kd3 = (1 + p3) ** (1 / 3), (1 - p3) ** (1 / 3)
    ku2, kd2 = np.sqrt(1.5), np.sqrt(0.5)        # 2D, P = 0.50
    cases = [
        ("3D, sfere secanti (tesi: Q0 = 1.076 dk)", 3, ku3, kd3, 0.076 * (ku3 - kd3)),
        ("3D, sfere tangenti (Q0 = dk)", 3, ku3, kd3, 0.0),
        ("2D, circonferenze tangenti (articolo MSCT: Q0 = dk)", 2, ku2, kd2, 0.0),
        ("2D, secanti come il nostro Q* (P = 0.50: dQ = 0.013)", 2, ku2, kd2, 0.013),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained", sharex=True)
    print(f"Delta = {a.delta}, gamma = {a.gamma}  (unita' E_F = kF = 1, eps = k^2)")
    for ax, (title, dim, ku, kd, dq) in zip(axs.flat, cases):
        ndn, nup, wc = case(dim, ku, kd, dq, a.delta, a.gamma, w)
        wdn, wup = w[np.argmin(ndn)], w[np.argmin(nup)]
        print(f"{title:52s} wc = {wc:.4f} | minimo N_dn a {wdn:+.4f}, N_up a {wup:+.4f}")
        ax.plot(w, ndn, color=C_DN, label="↓ minoritario")
        ax.plot(w, nup, color=C_UP, label="↑ maggioritario")
        for x, c in ((-wc, C_DN), (wc, C_UP)):
            ax.axvline(x, color=c, lw=0.8, ls="--")
        ax.set(title=f"{title}\n−ωc = {-wc:+.4f} (↓), +ωc = {wc:+.4f} (↑)", ylabel="N(ω) / N(ω; Δ = 0)")
        ax.legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("ω / E_F")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
