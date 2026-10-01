#!/usr/bin/env python3
"""Confronto delle funzioni in uscita fra run (o iterazioni) alla stessa P, dalle snapshot.

Per ogni gruppo (una colonna) sovrappone le run date:
  ReGamma^-1(Q,0) - max [delta]          (tabella pair completa)
  ImGamma^-1(qff, Omega) vicino a 0
  ReSigma_dn - sigma0 e ImSigma_dn a k ~ kF_dn   (righe salvate in sigmaNNN.npz)
  A_dn(kF_dn, omega) vicino a 0
  n_dn(k) (quella portata all'iterazione dopo)
Uso (dalla radice di SLIM):
  python3 test/compare_functions.py "P0p70_prod_union:3,P0p70_prod_union_fine:1" "P0p75_prod_union:3,P0p75_prod_union_fine:1"
      [--base out/cluster] [--out out/cluster/plots/compare_functions.png]
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
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ROWS = ["ReΓ⁻¹(Q,0) − max  [δ]", "ImΓ⁻¹(qff, Ω) ×10³", "ReΣ↓ − σ0 a kF↓", "ImΣ↓ a kF↓", "A↓(kF↓, ω)", "n↓(k)"]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="RUN:IT,RUN:IT,... (una colonna per gruppo)")
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--wlim", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "compare_functions.png"))
    a = ap.parse_args(argv)
    ncol = len(a.groups)
    fig, axs = plt.subplots(len(ROWS), ncol, figsize=(6.0 * ncol, 3.0 * len(ROWS)), layout="constrained", squeeze=False)
    for c, group in enumerate(a.groups):
        for n, spec in enumerate(group.split(",")):
            run, it = spec.split(":")
            it = int(it)
            col = CAT[n % len(CAT)]
            lab = f"{run} it {it}"
            z = np.load(os.path.join(a.base, run, "snap", f"iter{it:03d}.npz"))
            s = np.load(os.path.join(a.base, run, "snap", f"sigma{it:03d}.npz"))
            q, w, qff = np.asarray(z["q"]), np.asarray(z["omega"]), float(z["qff"])
            re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
            x = q / qff
            m = x <= 2.0
            axs[0, c].plot(x[m], (re0[m] - re0[m].max()) / 1e-3, "-o", ms=2.5, lw=1, color=col, label=lab)
            iq = int(np.argmin(np.abs(q - qff)))
            mw = np.abs(w) <= a.wlim
            axs[1, c].plot(w[mw], np.asarray(z["ImInvGamma"], float)[iq, mw] * 1e3, lw=1, color=col, label=lab)
            mu_dn = float(s["mu_dn_dn"])
            kf = np.sqrt(mu_dn)
            rk = np.asarray(s["rows_k_dn"], float)
            i = int(np.argmin(np.abs(rk - kf)))
            rw = np.asarray(s["rows_w_dn"][i], float)
            o = np.argsort(rw)
            rw = rw[o]
            mr = np.abs(rw) <= a.wlim
            s0 = float(s["sigma0_dn"])
            axs[2, c].plot(rw[mr], np.asarray(s["rows_ReS_dn"][i], float)[o][mr] - s0, lw=1, color=col, label=lab)
            axs[3, c].plot(rw[mr], np.asarray(s["rows_ImS_dn"][i], float)[o][mr], lw=1, color=col, label=lab)
            axs[4, c].plot(rw[mr], np.asarray(s["rows_A_dn"][i], float)[o][mr], lw=1, color=col, label=f"{lab} (k/kF={rk[i] / kf:.3f})")
            k = np.asarray(s["dr_k"], float)
            key = [kk for kk in s.files if kk.startswith("dr_alpha_") and kk.endswith("_nk_down") and kk != "dr_alpha_1_nk_down"]
            nk = np.asarray(s[key[0]] if key else s["dr_alpha_1_nk_down"], float)
            mk = k <= 3.0 * kf
            axs[5, c].plot(k[mk] / kf, nk[mk], lw=1, color=col, label=lab)
        P = run.split("_")[0].replace("P0p", "0.")
        for r, t in enumerate(ROWS):
            axs[r, c].set_title(f"P = {P}: {t}", fontsize=9)
            axs[r, c].legend(fontsize=7, frameon=False)
        axs[0, c].set_xlabel("Q / qff")
        for r in (1, 2, 3, 4):
            axs[r, c].set_xlabel("ω")
            axs[r, c].axvline(0, color="#6b6b68", lw=0.6)
        axs[5, c].set_xlabel("k / kF↓")
        axs[4, c].set_yscale("log")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
