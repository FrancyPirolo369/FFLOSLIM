#!/usr/bin/env python3
"""A/B della bolla coerente QPxQP: quadratura angolare di produzione (kjac) contro integrale nelle energie.

Fermioni liberi, eps = k^2 (m = 1/2), Z = 1.  Nelle variabili (E_up, E_dn) la misura e'
    int d^2k F(|k|, |Q-k|) = int dE_up dE_dn F / sqrt(R),
    R = ((k_up + Q)^2 - k_dn^2) (k_dn^2 - (k_up - Q)^2),   k_s^2 = E_s + kF_s^2
e con Gamma -> 0 la finestra di Pauli e' delta(Omega - E_up - E_dn) con E_up, E_dn fra 0 e Omega.
Lungo E_up + E_dn = Omega, R = 4 Q^2 (e + kF_up^2) - (2e + kF_up^2 - kF_dn^2 - Omega + Q^2)^2 e' un
polinomio di secondo grado in e = E_up, quindi
    ImPi(Q, Omega) = -sign(Omega) / (4 pi) * (1/2) [arcsin((2e - a - b) / (b - a))]_lo^hi
in forma chiusa (vuoto: -1/8).  Nessuna griglia: e' il riferimento esatto (A1 dell'articolo MSCT).

A: fflo.analytic_bubble.coherent_impi_kjac con i parametri di produzione (nk 96, 48 nodi angolari,
   larghezza 1e-3 per gamba), e con 160 nodi (gold).
B: la formula esatta.
S(Q) = (1/pi) int dOmega ImPi / Omega su |Omega| < W e' la parte di ReGamma^-1 con la cuspide.

Uso (dalla radice di SLIM):  python3 test/qpqp_ab_probe.py [--nquad 48 160]
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
C_EX, COLS = "#1f1f1e", ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5,
})


def impi_exact(Q, om, c1, c2):
    """ImPi_QP(Q, Omega) esatta per Gamma -> 0 (fermioni liberi, eps = k^2)."""
    om = np.asarray(om, float)
    out = np.zeros_like(om)
    for i, w in enumerate(om):
        # R(e) = -4 e^2 + B e + C
        d0 = c1 - c2 - w + Q * Q
        A, B, C = -4.0, 4.0 * Q * Q - 4.0 * d0, 4.0 * Q * Q * c1 - d0 * d0
        disc = B * B - 4 * A * C
        if disc <= 0:
            continue
        r1, r2 = sorted(((-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)))
        lo, hi = max(min(0.0, w), r1, -c1), min(max(0.0, w), r2, w + c2 if w > 0 else c2 + w + 1e9)
        # vincoli k_dn^2 = w - e + c2 >= 0  -> e <= w + c2
        hi = min(hi, w + c2)
        if hi <= lo:
            continue
        f = lambda e: np.arcsin(np.clip((2 * e - r1 - r2) / (r2 - r1), -1.0, 1.0))
        out[i] = -np.sign(w) / (4 * np.pi) * 0.5 * (f(hi) - f(lo))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=float, default=0.50)
    ap.add_argument("--nquad", type=int, nargs="+", default=[48, 160])
    ap.add_argument("--gamma", type=float, default=1e-3)
    ap.add_argument("--W", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "qpqp_ab_probe.png"))
    a = ap.parse_args(argv)
    kup, kdn = np.sqrt(1 + a.P), np.sqrt(1 - a.P)
    c1, c2, qff = kup ** 2, kdn ** 2, kup - kdn
    k = np.linspace(0.0, 4.0, 4001)
    Eu, Ed = k * k - c1, k * k - c2
    one, gam = np.ones_like(k), np.full_like(k, a.gamma)
    core = np.linspace(-0.02, 0.02, 80, endpoint=False) + 0.00025
    tail = np.geomspace(0.02, a.W, 60)[1:]
    om = np.unique(np.concatenate((-tail[::-1], core, tail)))
    x = np.arange(-0.03, 0.0601, 0.0025)

    # controllo del riferimento: vuoto -1/8 a Omega grande
    print(f"controllo: ImPi_esatta(qff, Omega = 50) = {impi_exact(qff, [50.0], c1, c2)[0]:+.6f} (atteso -0.125)")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    wz = om[np.abs(om) < 0.06]
    S_ex = np.array([np.trapezoid(impi_exact(qff * (1 + xv), om, c1, c2) / om, om) / np.pi for xv in x])
    i_ex = int(np.argmax(S_ex)) if S_ex.max() - S_ex[12] > abs(S_ex.min() - S_ex[12]) else int(np.argmin(S_ex))
    s0 = float(np.interp(0.0, x, S_ex))
    print(f"B esatta:        estremo a Q/qff = {1 + x[i_ex]:.4f}, S(Q*) - S(qff) = {S_ex[i_ex] - s0:+.3e}")
    axs[0].plot(wz, impi_exact(qff, wz, c1, c2), "-", color=C_EX, lw=2.2, label="esatta (energie, Γ → 0)")
    axs[1].plot(1 + x, S_ex - s0, "-", color=C_EX, lw=2.2, label="esatta (energie, Γ → 0)")
    for c, nq in zip(COLS, a.nquad):
        t0 = time.time()
        kw = dict(nk=96, n_kquad=nq, kquad_chunk_size=16, kmax=4.0, gamma_floor=a.gamma, kf_features=[kup, kdn])
        S = np.array([np.trapezoid(coherent_impi_kjac(qff * (1 + xv), om, k, Eu, one, gam, k, Ed, one, gam, **kw) / om, om)
                      / np.pi for xv in x])
        imq = coherent_impi_kjac(qff, wz, k, Eu, one, gam, k, Ed, one, gam, **kw)
        s0k = float(np.interp(0.0, x, S))
        dev = np.sqrt(np.mean(((S - s0k) - (S_ex - s0)) ** 2))
        i = int(np.argmax(S)) if S.max() - s0k > abs(S.min() - s0k) else int(np.argmin(S))
        print(f"A kjac n = {nq:4d}: estremo a Q/qff = {1 + x[i]:.4f}, S(Q*) - S(qff) = {S[i] - s0k:+.3e}, "
              f"scarto RMS dalla esatta = {dev:.2e}   [{time.time() - t0:.0f} s]")
        axs[0].plot(wz, imq, "o-", ms=2.5, lw=0.9, color=c, label=f"kjac, {nq} nodi angolari")
        axs[1].plot(1 + x, S - s0k, "o-", ms=2.5, lw=0.9, color=c, label=f"kjac, {nq} nodi angolari")
    axs[0].set(xlabel="Ω", ylabel="ImΠ_QP(qff, Ω)", title="ImΠ coerente alla tangenza, P = %.2f" % a.P)
    axs[1].axvline(1.0, color=MUTED, lw=0.8)
    axs[1].set(xlabel="Q / qff", ylabel="S(Q) − S(qff)", title="parte di ReΓ⁻¹ con la cuspide")
    for ax in axs:
        ax.legend(fontsize=7.5)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
