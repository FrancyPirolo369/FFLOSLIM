#!/usr/bin/env python3
"""Funzioni delle run ad alta P (Gamma, Sigma, n(k)) e controlli numerici, dalle snap.

Per ogni run, ultima iterazione (snap/iterNNN.npz e snap/sigmaNNN.npz):
  scale       g_c, eps0 = 2 exp(2 g) (= energia di legame a due corpi), sigma0 dei due spin
  densita'    n_s contro mu_s/2 (cubo emesso e fresca alpha = 1), quota di n_dn fuori dal mare di Fermi
  regola di somma  int A(k, w) dw - 1: massimo e dove
  Gamma       celle con ImGamma^-1 = 0 esatto a Omega != 0 (guardia di segno), righe spurie,
              rugosita' vicino a Q = 0
  contact     C della coppia contro la coda k^4 n_dn(k) a k = 6-8
  Sigma       righe salvate: ImSigma > 0 (segno sbagliato), dove sta il minimo di ImSigma_dn(kF, w)
Figura: ReGamma^-1_pin(Q,0), -ImGamma^-1 a Q = 0 e qff, n(k) dei due spin, k^4 n_dn(k),
-ImSigma_dn(kF, w), regola di somma per k.

Uso (dalla radice di SLIM):  python3 test/highP_functions_report.py [RUN ...]
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "out", "cluster")
DELTA = 1.0e-3
RUNS = ["P0p50_prod", "P0p70_gmax", "P0p75_gmax", "P0p75_prod_union_fine", "P0p80_x75a0p3",
        "P0p85_prod_union_fine", "P0p90_x75a0p3"]


def polar(run):
    return float("0." + run.split("_")[0][3:])


def last(run):
    it = sorted(glob.glob(os.path.join(BASE, run, "snap", "iter*.npz")))[-1]
    sg = it.replace("/iter", "/sigma")
    return np.load(it), (np.load(sg, allow_pickle=True) if os.path.exists(sg) else None), it[-7:-4]


def main(argv):
    runs = argv or RUNS
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for j, run in enumerate(runs):
        z, s, it = last(run)
        P = polar(run)
        c = cmap(j / max(1, len(runs) - 1))
        q, w, qff = z["q"].astype(float), z["omega"].astype(float), float(z["qff"])
        reI, imI = z["ReInvGamma"].astype(float), z["ImInvGamma"].astype(float)
        iw0 = int(np.argmin(np.abs(w)))
        sh = float(z["shift_used"])
        g = -4 * math.pi * sh - 0.5 * math.log(2)
        eps0 = 2 * math.exp(2 * g)
        pin = (reI[:, iw0] - sh - DELTA) / DELTA
        i0, iq = int(np.argmin(np.abs(q))), int(np.argmin(np.abs(q - qff)))
        lab = f"{run} it{it}"
        # --- controlli
        lines = [f"== {lab}: P={P}, g_c={g:+.3f}, eps0 = 2e^(2g) = {eps0:.2f}, C={float(z['pair_contact']):.3f}"]
        zero = (imI == 0.0) & (np.abs(w)[None, :] > 1e-9)
        qz = np.unique(np.round(q[np.any(zero, axis=1)] / qff, 2))
        lines.append(f"   Gamma: celle con ImG^-1 = 0 esatto a Omega != 0: {int(zero.sum())} "
                     f"(su {zero.size}); righe Q/qff toccate: {qz[:8]}{' ...' if qz.size > 8 else ''}")
        near0 = q / qff <= 0.12
        lines.append(f"   Gamma: gara pin(0) - pin(qff) = {pin[i0] - pin[iq]:+.2f} delta; rugosita' a Q~0 "
                     f"(std diff. seconde) {np.std(np.diff(pin[near0], 2)):.2f} delta; massimo in Q<=2qff a "
                     f"{q[int(np.argmax(np.where(q <= 2 * qff, pin, -np.inf)))] / qff:.3f} qff")
        if s is not None:
            for sp, mu in (("dn", 1 - P), ("up", 1 + P)):
                kk = np.asarray(s[f"k_{sp}"], float)
                kf = math.sqrt(mu)
                sa = np.asarray(s[f"sumA_{sp}"], float)
                m = kk <= 8
                ib = int(np.argmax(np.abs(sa[m] - 1)))
                n_emit = float(s[f"dr_alpha_0.3_density_{'down' if sp == 'dn' else 'up'}"])
                n_fresh = float(s[f"dr_alpha_1_density_{'down' if sp == 'dn' else 'up'}"])
                lines.append(f"   {sp}: sigma0 = {float(s[f'sigma0_{sp}']):+.3f} (mu = {mu:.2f}); n = {n_emit:.4f} "
                             f"emessa, {n_fresh:.4f} fresca (bersaglio {mu / 2:.4f}: {100 * (n_emit / (mu / 2) - 1):+.1f}% / "
                             f"{100 * (n_fresh / (mu / 2) - 1):+.1f}%); regola di somma: max |int A - 1| = "
                             f"{abs(sa[m][ib] - 1):.3f} a k/kF = {kk[m][ib] / kf:.2f}")
                ims = np.asarray(s[f"rows_ImS_{sp}"], float)
                lines.append(f"       righe salvate di Sigma: punti con ImSigma > 1e-6: {int((ims > 1e-6).sum())}")
            kd = np.asarray(s["k_dn"], float)
            nd = np.asarray(s["dr_alpha_0.3_nk_down"], float)
            tail = np.median((kd ** 4 * nd)[(kd >= 6) & (kd <= 7.9)])
            lines.append(f"   contact: C coppia {float(z['pair_contact']):.4f} contro k^4 n_dn(k) a k in [6, 8): {tail:.4f}")
            rk = np.asarray(s["rows_k_dn"], float)
            ir = int(np.argmin(np.abs(rk - math.sqrt(1 - P))))
            rw, rim = np.asarray(s["rows_w_dn"][ir], float), np.asarray(s["rows_ImS_dn"][ir], float)
            mm = (rw > -12) & (rw < 6)
            lines.append(f"   ImSigma_dn(kF, w): picco di -ImSigma a w = {rw[mm][int(np.argmin(rim[mm]))]:+.2f} "
                         f"(valore {-rim[mm].min():.2f}); a w = 0: {-np.interp(0, rw, rim):.4f}")
        print("\n".join(lines))
        # --- figura
        mq = q / qff <= 2.5
        axs[0, 0].plot(q[mq] / qff, np.clip(pin[mq], -40, 5), "-", color=c, lw=1.4, label=f"{lab}")
        for iqq, ls in ((i0, "-"), (iq, ":")):
            mw = (w > -4) & (w < 4)
            axs[0, 1].plot(w[mw], np.abs(imI[iqq, mw]), ls, color=c, lw=1.2)
        if s is not None:
            for sp, ls, mu in (("dn", "-", 1 - P), ("up", ":", 1 + P)):
                kk = np.asarray(s[f"k_{sp}"], float)
                axs[0, 2].plot(kk / math.sqrt(mu), np.asarray(s[f"dr_alpha_0.3_nk_{'down' if sp == 'dn' else 'up'}"], float),
                               ls, color=c, lw=1.3)
            kd = np.asarray(s["k_dn"], float)
            nd = np.asarray(s["dr_alpha_0.3_nk_down"], float)
            axs[1, 0].plot(kd, kd ** 4 * nd, "-", color=c, lw=1.3)
            axs[1, 0].axhline(float(z["pair_contact"]), color=c, lw=0.8, ls="--")
            rk = np.asarray(s["rows_k_dn"], float)
            ir = int(np.argmin(np.abs(rk - math.sqrt(1 - P))))
            rw, rim = np.asarray(s["rows_w_dn"][ir], float), np.asarray(s["rows_ImS_dn"][ir], float)
            mm = (rw > -12) & (rw < 6)
            axs[1, 1].plot(rw[mm], -rim[mm], "-", color=c, lw=1.3)
            sa = np.asarray(s["sumA_dn"], float)
            axs[1, 2].plot(kd / math.sqrt(1 - P), sa - 1, "-", color=c, lw=1.2)
    axs[0, 0].set(xlabel="Q / qff", ylabel="ReΓ⁻¹_pin(Q, 0) [δ]  (tagliata a −40)", title="ReΓ⁻¹(Q,0): pin = −1")
    axs[0, 0].legend(fontsize=7)
    axs[0, 1].set(xlabel="Ω", ylabel="|ImΓ⁻¹|", yscale="log", title="|ImΓ⁻¹| a Q = 0 (continua) e qff (punteggiata)")
    axs[0, 2].set(xlabel="k / kF_σ", ylabel="n(k)", xlim=(0, 4), title="n↓ (continua) e n↑ (punteggiata)")
    axs[1, 0].set(xlabel="k", ylabel="k⁴ n↓(k)", xlim=(0, 8.5), title="coda di contatto (tratteggio: C della coppia)")
    axs[1, 1].set(xlabel="ω", ylabel="−ImΣ↓(kF↓, ω)", title="ImΣ del minoritario a kF")
    axs[1, 2].set(xlabel="k / kF↓", ylabel="∫A↓ dω − 1", xlim=(0, 12), title="regola di somma del minoritario")
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    dst = os.path.join(HERE, "out", "cluster", "plots", "highP_functions.png")
    fig.savefig(dst, dpi=110)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
