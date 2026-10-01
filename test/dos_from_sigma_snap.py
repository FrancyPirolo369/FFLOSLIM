#!/usr/bin/env python3
"""N(omega) dalla snap di Sigma (snap/sigmaNNN.npz), senza le cube.

Sigma(k, w) = ReS0(k) [ReSigma(k, 0) sulla griglia fine a 406 k]
            + [Sigma_riga(w) - ReSigma_riga(0)] interpolato linearmente in k fra le 13 righe salvate
A(k, w) = (1/pi) g / ((w - (k^2 - mu) - (ReSigma - sigma0))^2 + g^2),  g = |ImSigma| + gamma_display
N(w) = int k dk A  (gas libero: 1/2).  Controllo: A ricostruita alle k delle righe contro rows_A salvate.

Uso (dalla radice di SLIM):  python3 test/dos_from_sigma_snap.py RUN [RUN ...] [--gamma 0.002]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e3df", "#fcfcfb"
COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def sigma_field(s, spin, w):
    """Sigma(k, w) come funzione: restituisce (k_fine, ReS(k_fine, w), ImS(k_fine, w)) e mu, sigma0."""
    rk = np.asarray(s[f"rows_k_{spin}"], float)
    dre, dim = [], []
    for i in range(rk.size):
        rw = np.asarray(s[f"rows_w_{spin}"][i], float)
        o = np.argsort(rw)
        rw, re, im = rw[o], np.asarray(s[f"rows_ReS_{spin}"][i], float)[o], np.asarray(s[f"rows_ImS_{spin}"][i], float)[o]
        dre.append(np.interp(w, rw, re) - np.interp(0.0, rw, re))
        dim.append(np.interp(w, rw, im))
    return rk, np.array(dre), np.array(dim)


def build(s, spin, w, k, gamma):
    mu = float(s[f"mu_{'up' if spin == 'up' else 'dn'}_{spin}"])
    s0 = float(s[f"sigma0_{spin}"])
    rk, dre, dim = sigma_field(s, spin, w)
    re0 = np.interp(k, np.asarray(s[f"k_{spin}"], float), np.asarray(s[f"reS0_{spin}"], float))
    # interpolazione lineare in k fra le righe, per ogni omega
    j = np.clip(np.searchsorted(rk, k) - 1, 0, rk.size - 2)
    t = np.clip((k - rk[j]) / (rk[j + 1] - rk[j]), 0.0, 1.0)
    re = re0[None, :] + (dre[j].T * (1 - t) + dre[j + 1].T * t)
    im = dim[j].T * (1 - t) + dim[j + 1].T * t
    g = np.abs(im) + gamma
    a = g / np.pi / ((w[:, None] - (k[None, :] ** 2 - mu) - (re - s0)) ** 2 + g ** 2)
    return a, mu, s0


def kgrid(kf_list, kmax=6.0):
    parts = [np.linspace(1e-4, kmax, 6000)]
    for kf in kf_list:
        parts.append(np.linspace(max(1e-4, kf - 0.15), kf + 0.15, 6000))
    return np.unique(np.concatenate(parts))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--gamma", type=float, default=0.002)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "dos_from_sigma_snap.png"))
    a = ap.parse_args(argv)
    wz = np.linspace(-0.3, 0.3, 601)
    ww = np.linspace(-1.5, 1.5, 601)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for c, run in zip(COLS, a.runs):
        path = sorted(glob.glob(os.path.join(a.base, run, "snap", "sigma*.npz")))[-1]
        s = np.load(path, allow_pickle=True)
        kfs = [np.sqrt(float(s["mu_up_up"])), np.sqrt(float(s["mu_dn_dn"]))]
        k = kgrid(kfs)
        for col, spin in ((0, "dn"), (1, "up")):
            # controllo: A ricostruita alle k delle righe contro rows_A
            rk = np.asarray(s[f"rows_k_{spin}"], float)
            i = int(np.argmin(np.abs(rk - np.sqrt(float(s[f"mu_{'up' if spin == 'up' else 'dn'}_{spin}"])))))
            rw = np.sort(np.asarray(s[f"rows_w_{spin}"][i], float))
            rw = rw[(rw > -0.3) & (rw < 0.3)]
            arec, mu, s0 = build(s, spin, rw, np.array([rk[i]]), 1e-3)  # eta della pipeline
            o = np.argsort(s[f"rows_w_{spin}"][i])
            asav = np.interp(rw, np.asarray(s[f"rows_w_{spin}"][i], float)[o], np.asarray(s[f"rows_A_{spin}"][i], float)[o])
            err = np.max(np.abs(arec[:, 0] - asav)) / np.max(asav)
            for row, w in ((0, wz), (1, ww)):
                aa, mu, s0 = build(s, spin, w, k, a.gamma)
                n = np.trapezoid(aa * k[None, :], k, axis=1)
                axs[row, col].plot(w, n, color=c, label=f"{run} ({os.path.basename(path)})")
                if row == 0:
                    i0 = int(np.argmin(np.abs(w)))
                    print(f"{run} {spin}: mu={mu:.3f} sigma0={s0:+.3f} kF={np.sqrt(mu):.3f}  N(0)={n[i0]:.3f}  "
                          f"N(-0.1)={np.interp(-0.1, w, n):.3f} N(+0.1)={np.interp(0.1, w, n):.3f}  "
                          f"controllo riga k={rk[i] / np.sqrt(mu):.3f} kF: max|A_ric - A_salvata|/max = {err:.2e}")
    for col, spin in ((0, "↓ minoritario"), (1, "↑ maggioritario")):
        axs[0, col].set(title=f"N(ω) {spin}, zoom", xlabel="ω", ylabel="N(ω)  (gas libero: 1/2)")
        axs[1, col].set(title=f"N(ω) {spin}, vista larga", xlabel="ω", ylabel="N(ω)")
        for ax in axs[:, col]:
            ax.axvline(0, color=MUTED, lw=0.6)
            ax.legend(fontsize=7)
    fig.suptitle(f"DOS ricostruita dalla snap di Σ (13 righe + ReΣ(k,0) fine), allargamento di disegno Γ = {a.gamma}", color=INK)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
