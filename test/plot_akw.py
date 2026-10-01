#!/usr/bin/env python3
"""Mappe A(k, omega) per i due spin e EDC a kF, dalle cube delle run (ultima next_cubes).

A ricostruita con Dyson da Sigma interpolata in k e omega (come test/plot_dos.py), allargamento
di visualizzazione Gamma piccolo; scala di colore logaritmica.  Righe = run, colonne = mappa up,
mappa down, EDC a kF (up e down).
Uso (dalla radice di SLIM):
  python3 test/plot_akw.py LABEL:RUN_DIR [...] [--gamma 0.005] [--kmax 2.2] [--wlo -1.2 --whi 0.6]
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
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cube_akw(path, k_out, w_out, gamma):
    c = np.load(path, allow_pickle=True)
    k = np.asarray(c["k"], float)
    wr = np.asarray(c["w"], float)
    re, im = np.asarray(c["ReS"], float), np.asarray(c["ImS"], float)
    spin_dn = "spindown" in os.path.basename(path)
    mu = float(c["mu_dn"] if spin_dn else c["mu_up"])
    mass = float(c["mass"]) if "mass" in c.files else 1.0
    s0 = float(c["sigma0"])
    wr = wr if wr.ndim == 2 else np.broadcast_to(wr, re.shape)
    reW = np.array([np.interp(w_out, wr[i], re[i]) for i in range(k.size)])
    imW = np.array([np.interp(w_out, wr[i], im[i]) for i in range(k.size)])
    j = np.clip(np.searchsorted(k, k_out, side="right") - 1, 0, k.size - 2)
    t = ((k_out - k[j]) / (k[j + 1] - k[j]))[:, None]
    reK = (1 - t) * reW[j] + t * reW[j + 1]
    imK = (1 - t) * imW[j] + t * imW[j + 1]
    g = np.abs(imK) + gamma
    xi = k_out[:, None] ** 2 / mass - mu
    return (1.0 / np.pi) * g / ((w_out[None, :] - xi - (reK - s0)) ** 2 + g ** 2), np.sqrt(mass * mu)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="LABEL:RUN_DIR")
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--kmax", type=float, default=2.2)
    ap.add_argument("--wlo", type=float, default=-1.2)
    ap.add_argument("--whi", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "akw.png"))
    a = ap.parse_args(argv)
    k_out = np.linspace(0.0, a.kmax, 900)
    w_out = np.linspace(a.wlo, a.whi, 900)
    n = len(a.sets)
    fig, axs = plt.subplots(n, 3, figsize=(15, 3.6 * n), layout="constrained", squeeze=False)
    for r, spec in enumerate(a.sets):
        label, run = spec.split(":", 1)
        up = sorted(glob.glob(os.path.join(run, "iter[0-9][0-9][0-9]", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
        dn = up.replace("spinup", "spindown")
        edc_w = np.linspace(a.wlo, a.whi, 3000)
        for c, (spin, path, col) in enumerate((("↑", up, "#2a78d6"), ("↓", dn, "#eb6834"))):
            A, kf = cube_akw(path, k_out, w_out, a.gamma)
            im = axs[r, c].pcolormesh(k_out, w_out, A.T, norm=LogNorm(vmin=1e-2, vmax=1e2), cmap="magma", shading="auto")
            axs[r, c].axvline(kf, color="w", lw=0.6, ls=":")
            axs[r, c].axhline(0, color="w", lw=0.6, ls=":")
            axs[r, c].set(title=f"{label}: A{spin}(k, ω)  (kF = {kf:.3f}, Γ = {a.gamma:g})", xlabel="k", ylabel="ω")
            fig.colorbar(im, ax=axs[r, c], shrink=0.85)
            Ae, _ = cube_akw(path, np.array([kf]), edc_w, a.gamma)
            axs[r, 2].semilogy(edc_w, Ae[0], lw=1, color=col, label=f"{spin} a kF")
        axs[r, 2].axvline(0, color="#6b6b68", lw=0.6)
        axs[r, 2].set(title=f"{label}: EDC a kF", xlabel="ω", ylabel="A(kF, ω)", ylim=(1e-2, 5e2))
        axs[r, 2].legend(fontsize=8, frameon=False)
        print(f"{label}: {os.path.relpath(up, HERE)}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
