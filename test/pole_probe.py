#!/usr/bin/env python3
"""Quadratura del polo di quasiparticella vicino a kF: riga attuale contro due cure, sulle cube salvate.

Per ogni riga k con |k - kF| < band*kF (spin dato) si calcolano sum A = int A dw e n(k) = int_{w<=0} A dw
con la stessa Sigma (PCHIP dai nodi della riga) e la stessa A della cube,
    A = (1/pi) g / ((w - xi_k - (ReS - sigma0))^2 + g^2),   g = |ImS| + eta,
in quattro modi:
  row   la riga della cube cosi' com'e' (quello che fa density.py oggi)
  zlog  riga con spalle logaritmiche attorno al polo (fflo.density._two_pole_omega_row), stessi nodi
  sub   riga attuale, ma si toglie L = Z/pi * Zg / ((w - E)^2 + (Zg)^2) e se ne somma l'integrale esatto
  ref   griglia ultra-fitta (passo 5e-8 attorno al polo)
Stampa gli scarti da ref e l'effetto integrato sulla densita' (int k n(k) dk nella banda, in % di mu/2).

Uso (dalla radice di SLIM):
  python3 test/pole_probe.py [RUN ...] [--spin down] [--band 0.3] [--out out/impi_probe/pole_probe]
  RUN = cartella con iterNNN/next_cubes (default: P0p75_gmax P0p65_prod P0p50_prod P0p90_x75a0p3)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from fflo.density import _two_pole_omega_row  # noqa: E402


def a_of(w, re_f, im_f, xi, s0, eta):
    g = np.abs(im_f(w)) + eta
    return (1.0 / np.pi) * g / ((w - xi - (re_f(w) - s0)) ** 2 + g ** 2)


def split(w, a):
    """(totale, parte w <= 0); la riga deve contenere w = 0."""
    neg = w <= 0.0
    return np.trapezoid(a, w), np.trapezoid(a[neg], w[neg])


def with_zero(w):
    return np.unique(np.concatenate([w, [0.0]]))


def probe(cube, band):
    c = np.load(cube, allow_pickle=True)
    k, W = np.asarray(c["k"], float), np.asarray(c["w"], float)
    RE, IM = np.asarray(c["ReS"], float), np.asarray(c["ImS"], float)
    spin_dn = "spindown" in os.path.basename(cube)
    mu = float(c["mu_dn"] if spin_dn else c["mu_up"])
    mass = float(c["mass"]) if "mass" in c.files else 1.0
    s0, eta = float(c["sigma0"]), float(c["eta"])
    w_base = np.asarray(c["w_base"], float).reshape(-1) if "w_base" in c.files else W[0]
    kf = np.sqrt(mass * mu)
    W = W if W.ndim == 2 else np.broadcast_to(W, RE.shape)
    out = []
    for j in np.flatnonzero(np.abs(k - kf) < band * kf):
        wr = W[j]
        re_f, im_f = PchipInterpolator(wr, RE[j]), PchipInterpolator(wr, IM[j])
        xi = k[j] ** 2 / mass - mu
        # polo: zero del residuo di Dyson, quello con A piu' grande
        f = wr - xi - (RE[j] - s0)
        sc = np.flatnonzero(np.diff(np.sign(f)) != 0)
        if sc.size == 0:
            continue
        roots = [wr[i] - f[i] * (wr[i + 1] - wr[i]) / (f[i + 1] - f[i]) for i in sc]
        E = max(roots, key=lambda x: float(a_of(np.array([x]), re_f, im_f, xi, s0, eta)[0]))
        Z = 1.0 / (1.0 - float(re_f.derivative()(E)))
        gam = Z * (abs(float(im_f(E))) + eta)
        # row: riga attuale
        w0 = with_zero(wr)
        tot_row, n_row = split(w0, a_of(w0, re_f, im_f, xi, s0, eta))
        # zlog: spalle logaritmiche attorno al polo, stesso numero di nodi della riga
        wz = with_zero(_two_pole_omega_row(w_base, E, gam, np.nan, 0.0, 121))
        tot_z, n_z = split(wz, a_of(wz, re_f, im_f, xi, s0, eta))
        # sub: sottrazione analitica della lorentziana sulla riga attuale
        L = (Z / np.pi) * gam / ((w0 - E) ** 2 + gam ** 2)
        t_rem, n_rem = split(w0, a_of(w0, re_f, im_f, xi, s0, eta) - L)
        tot_s = t_rem + Z * (0.5 + np.arctan((w0[-1] - E) / gam) / np.pi - (0.5 + np.arctan((w0[0] - E) / gam) / np.pi))
        n_s = n_rem + Z * (np.arctan((0.0 - E) / gam) - np.arctan((w0[0] - E) / gam)) / np.pi
        # ref: ultra-fitta
        wf = np.unique(np.concatenate([
            np.linspace(wr[0], -0.5, 20001), np.linspace(-0.5, 0.5, 400001),
            np.linspace(E - 2e-3, E + 2e-3, 80001), np.linspace(0.5, wr[-1], 20001), [0.0]]))
        tot_f, n_f = split(wf, a_of(wf, re_f, im_f, xi, s0, eta))
        out.append((k[j], E, Z, gam, tot_row, tot_z, tot_s, tot_f, n_row, n_z, n_s, n_f))
    return np.array(out), kf, mu


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=["P0p75_gmax", "P0p65_prod", "P0p50_prod", "P0p90_x75a0p3"])
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--spin", choices=("up", "down"), default="down")
    ap.add_argument("--band", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "impi_probe", "pole_probe"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    runs = [r for r in a.runs if glob.glob(os.path.join(a.base, r, "iter*", "next_cubes"))]
    fig, axs = plt.subplots(2, len(runs), figsize=(4.6 * len(runs), 7), layout="constrained", squeeze=False)
    for c, run in enumerate(runs):
        cube = sorted(glob.glob(os.path.join(a.base, run, "iter*", "next_cubes", f"A_komega_spin{a.spin}_iter*.npz")))[-1]
        r, kf, mu = probe(cube, a.band)
        k = r[:, 0]
        print(f"== {run} ({os.path.basename(cube)})  kF = {kf:.3f}  righe {len(r)}  Z a kF = {r[np.argmin(abs(k - kf)), 2]:.3f}")
        for name, i in (("row ", 4), ("zlog", 5), ("sub ", 6)):
            ds, dn = r[:, i] - r[:, 7], r[:, i + 4] - r[:, 11]
            dens = np.trapezoid(k * dn, k) / (mu / 2)
            print(f"   {name}: max|sumA - ref| {np.max(np.abs(ds)):.4f}   max|n(k) - ref| {np.max(np.abs(dn)):.4f}   "
                  f"densita' nella banda {100 * dens:+.3f}% di mu/2")
        print(f"   ref : sumA fra {r[:, 7].min():.4f} e {r[:, 7].max():.4f}")
        for name, i, col in (("riga attuale", 0, "#eb6834"), ("spalle log", 1, "#2a78d6"),
                             ("sottrazione", 2, "#1baf7a"), ("ref fitta", 3, "#1f1f1e")):
            ls = "--" if i == 3 else "-"
            axs[0, c].plot(k / kf, r[:, 4 + i], ls, lw=1, color=col, label=name)
            axs[1, c].plot(k / kf, r[:, 8 + i], ls, lw=1, color=col, label=name)
        axs[0, c].set(title=f"{run}: ∫A dω", xlabel="k / kF")
        axs[1, c].set(title=f"{run}: n(k)", xlabel="k / kF")
        axs[0, c].legend(fontsize=7, frameon=False)
    path = os.path.join(a.out, f"pole_probe_{a.spin}.png")
    fig.savefig(path, dpi=120)
    print("scritto", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
