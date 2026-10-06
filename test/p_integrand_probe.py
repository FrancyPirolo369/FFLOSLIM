#!/usr/bin/env python3
"""L'integrando radiale in p della parte residua di ImPi, visto da vicino.

Per qualche (Q, Omega) si calcola, con lo stesso codice di impi_table (test/impi_union_probe.py: A
ricostruita da Sigma, integrale angolare phipanel, integrale in eps a pannelli sul reticolo),
la funzione  f(p) = p * int deps K(eps, Omega) [A_up A_dn - A0_up A0_dn](p, ...)  su una griglia in p
FITTA, e la si confronta con i 31 campioni della griglia uniforme di produzione: gli spigoli dove i
gusci di Fermi diventano tangenti (p = kF_up, |Q +- kF_dn|) cadono fra un nodo e l'altro.

Uso (dalla radice di SLIM):  python3 test/p_integrand_probe.py [RUN] [--np-fine 1201]
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
import impi_union_probe as U  # noqa: E402
import fflo.impi_table as T  # noqa: E402
from fflo.adn_phi import build_phipanel_geometry_cache, precompute_kjac_integrated_slice  # noqa: E402
from fflo.kramers_kronig import _thermal_kernel_scalar_zero_temp, _trapz_axis_panel_kernel  # noqa: E402

COLS = ["#2a78d6", "#eb6834", "#1baf7a"]


def integrand(qv, om):
    G = U.G
    p = G["p"]
    eps = G["lat"]
    geom = build_phipanel_geometry_cache(G["kd"], p, q_value=float(qv), routing="difference",
                                         n_phi_panel=G["n_kquad"], phi_panel_chunk_size=4,
                                         k_features=G["kf"], k_feature_n_local=G["kquad_feat"],
                                         k_feature_half_width=0.05)
    aup, inu = T._rebuild_a_bilinear(p[:, None], eps[None, :], G["ku"], G["wu"], G["re_up"], G["im_up"],
                                     mu=G["mu_up"], sigma0=G["s0u"], eta=G["etau"])
    cup, inc = T.bilinear_eval_with_support(p[:, None], eps[None, :], G["cku"], G["cwu"], G["cau"])
    aup, cup = np.where(inu, aup, 0.0), np.where(inc, cup, 0.0)
    full, _ = precompute_kjac_integrated_slice(
        G["kd"], G["wd"], G["ad"], p, eps, q_value=float(qv), omega_value=float(om), routing="difference",
        n_kquad=G["n_kquad"], kquad_chunk_size=4, geom_cache=geom, re_sigma=G["re_dn"],
        im_sigma=G["im_dn"], sigma0=G["s0d"], mu=G["mu_dn"], eta=G["etad"])
    ctrl, _ = precompute_kjac_integrated_slice(
        G["ckd"], G["cwd"], G["cad"], p, eps, q_value=float(qv), omega_value=float(om),
        routing="difference", n_kquad=G["n_kquad"], kquad_chunk_size=4, geom_cache=geom)
    prod = aup * full - cup * ctrl
    i_p = _trapz_axis_panel_kernel(prod, eps, float(om), _thermal_kernel_scalar_zero_temp)
    return p * i_p


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", default="P0p75_gmax")
    ap.add_argument("--np-fine", type=int, default=1201)
    ap.add_argument("--dw", type=float, default=2.5e-4)
    a = ap.parse_args(argv)
    up = sorted(glob.glob(os.path.join(HERE, "out", "cluster", a.run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    work = os.path.join(HERE, "out", "impi_probe", "highP_integration", a.run)
    qff = U.setup(up, dn, os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz"),
                  n_p=a.np_fine, dw=a.dw, n_linear=int(round(40 * 1e-3 / a.dw)))
    kfu, kfd = U.G["kf"]
    p = U.G["p"]
    p31 = np.linspace(0.0, 4.0, 31)
    cases = [(1.0, 0.3), (1.0, -0.05), (0.014, 0.3), (0.0, 0.3)]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.4), layout="constrained")
    for ax, (qr, om) in zip(axs.flat, cases):
        qv = qr * qff
        f = integrand(qv, om)
        f31 = np.interp(p31, p, f)
        ref = np.trapezoid(f, p)
        tr31 = np.trapezoid(f31, p31)
        ax.plot(p, f, color="#1f1f1e", lw=1.2, label=f"griglia fitta ({p.size} nodi): ∫ = {ref:+.4e}")
        ax.plot(p31, f31, "o-", color=COLS[1], ms=4, lw=1.0, alpha=0.9,
                label=f"31 nodi uniformi (prod): ∫ = {tr31:+.4e}  ({100 * (tr31 / ref - 1):+.1f}%)")
        for x, lab in ((kfu, "kF↑"), (qv + kfd, "Q+kF↓"), (abs(qv - kfd), "|Q−kF↓|")):
            ax.axvline(x, color=COLS[0], lw=0.8, ls="--")
            ax.text(x, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0, f" {lab}", color=COLS[0], fontsize=7, va="top")
        ax.set(xlim=(0, 2.6), xlabel="p = |k↑|", ylabel="p · ∫dε K [A↑A↓ − A0↑A0↓]",
               title=f"{a.run}: Q = {qr:g} qff, Ω = {om:+g}")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
        print(f"Q = {qr:g} qff, Omega = {om:+g}: fitta {ref:+.5e}, 31 nodi {tr31:+.5e} ({100 * (tr31 / ref - 1):+.2f}%)", flush=True)
    dst = os.path.join(HERE, "out", "cluster", "plots", f"p_integrand_{a.run}.png")
    fig.savefig(dst, dpi=115)
    print("scritto", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
