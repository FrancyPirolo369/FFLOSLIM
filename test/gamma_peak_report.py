#!/usr/bin/env python3
"""Il picco di Gamma vicino a (qff, Omega = 0): forma, regime critico e come viene integrato.

Dalle snap (snap/iterNNN.npz: tabella di coppia GREZZA dell'iterazione) si rifa' il pin di density
(apply_shift_to_pair_table, delta = 1e-3, exact_zero) e si controlla che il contact torni uguale
a quello del log.  Poi:
  1. ReGamma_pin^-1(Q, 0) in unita' di delta attorno a qff (la cuspide; D e Q*/qff)
  2. -ImGamma^-1(qff, Omega) contro |Omega| in log-log: esponente vicino a 0
  3. A_pair(qff, Omega) sui nodi della tabella contro la stessa ricostruita su una griglia 20x piu'
     fitta interpolando Gamma^-1 (liscia) invece di A (a punta)
  4. il contact: da dove viene (regioni in Q e Omega) e quanto cambia integrando il picco cosi'
     (stessa ricetta di density.pair_contact_from_shifted_table, trapezio su q e Omega <= 0)

Uso (dalla radice di SLIM):  python3 test/gamma_peak_report.py [RUN:IT ...]
    default: P0p50_abl_T0_ctrl:8 P0p50_abl_R8_qff:1 P0p50_abl_R8_qff:6
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from fflo.density import pair_contact_from_shifted_table  # noqa: E402
from fflo.pair_shifted import apply_shift_to_pair_table  # noqa: E402

DELTA = 1.0e-3
COLS = ["#1f1f1e", "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
QWIN = (0.85, 1.15)        # finestra in Q/qff dove si rifinisce
WWIN = (-0.06, 0.0)        # finestra in Omega (solo Omega <= 0 entra nel contact)
REFINE = 20


def load(run, it):
    z = np.load(os.path.join(HERE, "out", "cluster", run, "snap", f"iter{it:03d}.npz"))
    q, w, qff = z["q"].astype(float), z["omega"].astype(float), float(z["qff"])
    t = {"q": q, "omega": w, "ReInvGamma": z["ReInvGamma"].astype(float),
         "ImInvGamma": z["ImInvGamma"].astype(float)}
    qs = float(q[np.argmin(np.abs(q - qff))])
    sh = apply_shift_to_pair_table(t, qs, subcritical_delta=DELTA, eta_floor_mode="exact_zero")
    re = np.asarray(sh["ReInvGamma"], float)       # gia' pinnata: ReGamma_pin^-1(q_sel, 0) = -delta
    im = np.asarray(sh["ImInvGamma"], float)
    return dict(q=q, w=w, qff=qff, re=re, im=im, a=np.asarray(sh["A_pair"], float),
                C_log=float(z["pair_contact"]))


def a_from(re, im):
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (1.0 / np.pi) * im / (re * re + im * im)      # A = -(1/pi) Im 1/(re + i im)
    return np.where(np.isfinite(a), a, 0.0)


def contact_parts(d):
    """C totale, C dalla finestra del picco (trapezio sui nodi) e la stessa rifinita."""
    q, w, a = d["q"], d["w"], d["a"]
    _, c_all = pair_contact_from_shifted_table(q, w, a)
    mq = (q / d["qff"] >= QWIN[0]) & (q / d["qff"] <= QWIN[1])
    mw = (w >= WWIN[0]) & (w <= WWIN[1])
    qn, wn = q[mq], w[mw]

    def c_box(qq, ww, aa):
        return -np.trapezoid(qq * np.trapezoid(aa, ww, axis=1), qq) / (2 * np.pi) / 4.0

    c_nodes = c_box(qn, wn, a[np.ix_(mq, mw)])
    # griglia fitta: Gamma^-1 interpolata linearmente (in Q e in Omega) fra i nodi della tabella
    qf = np.unique(np.concatenate([np.linspace(a_, b_, REFINE + 1) for a_, b_ in zip(qn[:-1], qn[1:])]))
    wf = np.unique(np.concatenate([np.linspace(a_, b_, REFINE + 1) for a_, b_ in zip(wn[:-1], wn[1:])]))

    def interp2(f):
        g = np.array([np.interp(wf, wn, row) for row in f[np.ix_(mq, mw)]])
        return np.array([np.interp(qf, qn, col) for col in g.T]).T

    re_f, im_f = interp2(d["re"]), interp2(d["im"])
    c_fine = c_box(qf, wf, a_from(re_f, im_f))
    return c_all, c_nodes, c_fine, (qf, wf, re_f, im_f)


def main(argv):
    specs = argv or ["P0p50_abl_T0_ctrl:8", "P0p50_abl_R8_qff:1", "P0p50_abl_R8_qff:6"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), layout="constrained")
    print(f"finestra del picco: Q/qff in {QWIN}, Omega in {WWIN}; rifinitura x{REFINE} interpolando Gamma^-1")
    for c, spec in zip(COLS, specs):
        run, it = spec.split(":")
        d = load(run, int(it))
        lab = f"{run.replace('P0p50_abl_', '')} it{int(it)}"
        q, w, qff = d["q"], d["w"], d["qff"]
        iw0 = int(np.argmin(np.abs(w)))
        iq = int(np.argmin(np.abs(q - qff)))
        c_all, c_nodes, c_fine, (qf, wf, re_f, im_f) = contact_parts(d)
        # 1. cuspide in Q
        r0 = d["re"][:, iw0] / DELTA
        m = (q / qff > 0.8) & (q / qff < 1.25)
        axs[0, 0].plot(q[m] / qff, r0[m], "o-", color=c, ms=3.5, lw=1.4, label=lab)
        imax = int(np.argmax(np.where(q <= 2 * qff, r0, -np.inf)))
        # 2. ImGamma^-1(qff, Omega)
        neg = (w < 0) & (w > -0.2)
        pos = (w > 0) & (w < 0.2)
        axs[0, 1].loglog(-w[neg], np.abs(d["im"][iq, neg]), "-", color=c, lw=1.4, label=f"{lab}, Ω<0")
        axs[0, 1].loglog(w[pos], np.abs(d["im"][iq, pos]), ":", color=c, lw=1.2)
        fit = (w < -0.002) & (w > -0.05) & (np.abs(d["im"][iq]) > 0)
        beta = np.polyfit(np.log(-w[fit]), np.log(np.abs(d["im"][iq, fit])), 1)[0]
        nz = int(np.sum(((w < 0) & (w > -0.05)) & (d["im"][iq] == 0)))
        # 3. A_pair(qff, Omega): nodi contro griglia fitta
        mm = (w > -0.03) & (w <= 0.0)
        axs[1, 0].plot(w[mm], -d["a"][iq, mm], "o", color=c, ms=4, label=f"{lab}: nodi della tabella")
        jq = int(np.argmin(np.abs(qf - q[iq])))
        mf = wf > -0.03
        axs[1, 0].plot(wf[mf], -a_from(re_f[jq, mf], im_f[jq, mf]), "-", color=c, lw=1.2,
                       label=f"{lab}: Γ⁻¹ interpolata (x{REFINE})")
        # 4. contributo al contact per Q
        mw_ = w <= 0
        per_q = -q * np.trapezoid(d["a"][:, mw_], w[mw_], axis=1) / (2 * np.pi) / 4.0
        axs[1, 1].plot(q / qff, per_q, "-", color=c, lw=1.4, label=lab)
        print(f"{lab:16s} C = {c_all:.6f} (log {d['C_log']:.6f}) | D = {r0[imax] - r0[iq]:.2f} delta a Q* = "
              f"{q[imax] / qff:.3f} qff | -ImG^-1(qff,Omega<0) ~ |Omega|^{beta:.2f} ({nz} nodi a 0) | "
              f"C nella finestra: nodi {c_nodes:.5f} ({100 * c_nodes / c_all:.1f}% di C), "
              f"rifinita {c_fine:.5f}  ->  dC = {c_fine - c_nodes:+.5f} ({100 * (c_fine - c_nodes) / c_all:+.2f}% di C)")
        mw_ = w <= 0
        cell = -np.trapezoid(q[:, None] * np.where(mw_[None, :], d["a"], 0.0), q, axis=0) / (2 * np.pi) / 4.0
        parts = []
        for lo, hi in ((0.0, 0.06), (0.06, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 1e9)):
            m_ = mw_ & (-w >= lo) & (-w < hi)
            parts.append(f"|Ω| in [{lo:g},{hi:g}): {100 * np.trapezoid(np.where(m_, cell, 0.0), w) / c_all:5.1f}%")
        print("      contact per |Omega|:  " + "  ".join(parts))
        per = -q * np.trapezoid(d["a"][:, mw_], w[mw_], axis=1) / (2 * np.pi) / 4.0
        parts = []
        for lo, hi in ((0, 0.5), (0.5, 0.85), (0.85, 1.15), (1.15, 2.0), (2.0, 4.0), (4.0, 1e9)):
            m_ = (q / qff >= lo) & (q / qff < hi)
            parts.append(f"Q/qff in [{lo:g},{hi:g}): {100 * np.trapezoid(np.where(m_, per, 0.0), q) / c_all:5.1f}%")
        print("      contact per Q:        " + "  ".join(parts))
    for x in (2 / 3, 1.0):
        xs = np.array([1e-3, 0.1])
        axs[0, 1].loglog(xs, 0.02 * (xs / 0.01) ** x, color="#9a9a96", ls="--" if x < 1 else "-.", lw=1,
                         label=f"|Ω|^{x:.2g}")
    axs[0, 0].axvline(1.0, color="#9a9a96", lw=0.8)
    axs[0, 0].set(xlabel="Q/qff", ylabel="ReΓ⁻¹_pin(Q, 0) / δ", title="cuspide in Q (pin a qff: −1)")
    axs[0, 1].set(xlabel="|Ω|", ylabel="|ImΓ⁻¹(qff, Ω)|", title="regime critico a qff (continua Ω<0, punteggiata Ω>0)")
    axs[1, 0].set(xlabel="Ω", ylabel="−A_pair(qff, Ω)", title="il picco a qff: nodi della tabella contro Γ⁻¹ interpolata")
    axs[1, 1].set(xlabel="Q/qff", ylabel="contributo al contact per Q", xlim=(0, 4),
                  title="da dove viene il contact (−q ∫_{Ω≤0} A dΩ /8π)")
    for ax in axs.flat:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    dst = os.path.join(HERE, "out", "cluster", "plots", "gamma_peak_report.png")
    fig.savefig(dst, dpi=115)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
