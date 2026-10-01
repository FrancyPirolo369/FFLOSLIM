#!/usr/bin/env python3
"""Lo spike di N(omega) in omega = 0 viene dalla cuspide non-FL della riga di Sigma a kF spalmata in k?

Stessa ricostruzione di test/dos_from_sigma_snap.py (ReSigma(k,0) sulla griglia fine + forma in omega
interpolata linearmente in k fra le righe salvate), in tre versioni:
  base       : come la pipeline
  senza_kF   : la forma in omega della riga a kF e' sostituita dall'interpolazione fra le righe vicine
               (niente cuspide non-FL): se lo spike e' questo meccanismo, deve sparire
  confinata  : si aggiungono righe sintetiche a kF +- dk con la forma liscia delle vicine, cosi' la
               cuspide della riga a kF resta confinata entro dk (quello che farebbe una griglia in k fitta
               vicino a kF): lo spike deve restringersi/abbassarsi al calare di dk

Uso (dalla radice di SLIM):  python3 test/dos_spike_mechanism.py [SIGMA_SNAP] [--dk 0.05 0.02]
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
sys.path.insert(0, os.path.join(HERE, "test"))
from dos_from_sigma_snap import kgrid, sigma_field  # noqa: E402

COLS = ["#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def dos(s, spin, w, k, gamma, rk, dre, dim):
    mu = float(s[f"mu_{'up' if spin == 'up' else 'dn'}_{spin}"])
    s0 = float(s[f"sigma0_{spin}"])
    re0 = np.interp(k, np.asarray(s[f"k_{spin}"], float), np.asarray(s[f"reS0_{spin}"], float))
    j = np.clip(np.searchsorted(rk, k) - 1, 0, rk.size - 2)
    t = np.clip((k - rk[j]) / (rk[j + 1] - rk[j]), 0.0, 1.0)
    re = re0[None, :] + (dre[j].T * (1 - t) + dre[j + 1].T * t)
    im = dim[j].T * (1 - t) + dim[j + 1].T * t
    g = np.abs(im) + gamma
    a = g / np.pi / ((w[:, None] - (k[None, :] ** 2 - mu) - (re - s0)) ** 2 + g ** 2)
    return np.trapezoid(a * k[None, :], k, axis=1)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snap", nargs="?", default=os.path.join(HERE, "out", "cluster", "P0p50_abl_R8_qff", "snap", "sigma001.npz"))
    ap.add_argument("--dk", type=float, nargs="+", default=[0.05, 0.02])
    ap.add_argument("--gamma", type=float, default=0.002)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "dos_spike_mechanism.png"))
    a = ap.parse_args(argv)
    s = np.load(a.snap, allow_pickle=True)
    w = np.linspace(-0.08, 0.08, 641)
    k = kgrid([np.sqrt(float(s["mu_up_up"])), np.sqrt(float(s["mu_dn_dn"]))])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for col, spin in ((0, "dn"), (1, "up")):
        kf = np.sqrt(float(s[f"mu_{'up' if spin == 'up' else 'dn'}_{spin}"]))
        rk, dre, dim = sigma_field(s, spin, w)
        i = int(np.argmin(np.abs(rk - kf)))
        f = (rk[i] - rk[i - 1]) / (rk[i + 1] - rk[i - 1])
        smooth_re = (1 - f) * dre[i - 1] + f * dre[i + 1]
        smooth_im = (1 - f) * dim[i - 1] + f * dim[i + 1]
        versions = [("base (pipeline)", rk, dre, dim)]
        d2, m2 = dre.copy(), dim.copy()
        d2[i], m2[i] = smooth_re, smooth_im
        versions.append(("senza cuspide alla riga kF", rk, d2, m2))
        for dk in a.dk:
            add = np.array([rk[i] - dk * kf, rk[i] + dk * kf])
            wts = [(x - rk[i - 1]) / (rk[i + 1] - rk[i - 1]) for x in add]
            rk3 = np.sort(np.concatenate((rk, add)))
            d3 = np.array([dre[np.flatnonzero(rk == x)[0]] if x in rk else (1 - wts[list(add).index(x)]) * dre[i - 1] + wts[list(add).index(x)] * dre[i + 1] for x in rk3])
            m3 = np.array([dim[np.flatnonzero(rk == x)[0]] if x in rk else (1 - wts[list(add).index(x)]) * dim[i - 1] + wts[list(add).index(x)] * dim[i + 1] for x in rk3])
            versions.append((f"cuspide confinata a kF ± {dk:g} kF", rk3, d3, m3))
        i0 = int(np.argmin(np.abs(w)))
        for c, (lab, r_, d_, m_) in zip(COLS, versions):
            n = dos(s, spin, w, k, a.gamma, r_, d_, m_)
            side = 0.5 * (np.interp(-0.02, w, n) + np.interp(0.02, w, n))
            print(f"{spin} {lab:34s}: N(0) = {n[i0]:.4f}, N(+-0.02) = {side:.4f}, spike = {n[i0] - side:+.4f}")
            axs[col].plot(w, n, color=c, label=lab)
        axs[col].set(title=f"N({'↓' if spin == 'dn' else '↑'}) — {os.path.basename(os.path.dirname(os.path.dirname(a.snap)))}", xlabel="ω")
        axs[col].legend(fontsize=7)
        axs[col].grid(alpha=0.3)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
