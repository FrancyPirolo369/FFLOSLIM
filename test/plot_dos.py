#!/usr/bin/env python3
"""N(omega) = int k dk A(k, omega) per spin (asintoto libero 1/2), dalle cube A_komega di SLIM.

Ricetta (nota project-dos-pseudogap): le cube hanno eta = 1e-3, quindi il polo a kF e' quasi
una delta e la somma sui nodi in k da' una DOS a punte.  Si ricostruisce A con Dyson da Sigma
su una griglia in k piu' fitta (Sigma e' liscia in k) con un allargamento di visualizzazione
Gamma (default 0.02: fonde i poli senza riempire la pseudogap; 0.1 la mangia):
    A = (1/pi) g / ((w - xi_k - (ReS - sigma0))^2 + g^2),  g = |ImS| + Gamma.
Stampa anche N(0)/N_libera e la densita' int_{w<0} N dw contro mu/2.

Uso (dalla radice di SLIM):
  python3 test/plot_dos.py LABEL:UP.npz:DOWN.npz | RUN_DIR | LABEL:RUN_DIR  [...] [--gamma 0.02] [--wlim 3]
  (RUN_DIR = out/cluster/P0p50_prod ecc.: usa l'ultima iterazione con le next_cubes)
      [--kref 8] [--kmax 8] [--out out/cluster/plots/dos.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def dos(path, w, gamma, kref, kmax):
    c = np.load(path, allow_pickle=True)
    k = np.asarray(c["k"], float)
    wr = np.asarray(c["w"], float)
    re, im = np.asarray(c["ReS"], float), np.asarray(c["ImS"], float)
    spin_dn = "spindown" in os.path.basename(path)
    mu = float(c["mu_dn"] if spin_dn and "mu_dn" in c.files else (c["mu_down"] if spin_dn else c["mu_up"]))
    mass = float(c["mass"]) if "mass" in c.files else 1.0
    s0 = float(c["sigma0"])
    keep = k <= kmax
    k, re, im = k[keep], re[keep], im[keep]
    wr = wr[keep] if wr.ndim == 2 else np.broadcast_to(wr, re.shape)
    # Sigma su una griglia omega comune, riga per riga
    reW = np.array([np.interp(w, wr[i], re[i]) for i in range(k.size)])
    imW = np.array([np.interp(w, wr[i], im[i]) for i in range(k.size)])
    # k piu' fitta: kref nodi per intervallo
    kf = np.concatenate([np.linspace(k[i], k[i + 1], kref, endpoint=False) for i in range(k.size - 1)] + [k[-1:]])
    j = np.clip(np.searchsorted(k, kf, side="right") - 1, 0, k.size - 2)
    t = ((kf - k[j]) / (k[j + 1] - k[j]))[:, None]
    reK = (1 - t) * reW[j] + t * reW[j + 1]
    imK = (1 - t) * imW[j] + t * imW[j + 1]
    g = np.abs(imK) + gamma
    xi = kf[:, None] ** 2 / mass - mu
    A = (1.0 / np.pi) * g / ((w[None, :] - xi - (reK - s0)) ** 2 + g ** 2)
    N = np.trapezoid(kf[:, None] * A, kf, axis=0)
    return N, mu, mass


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="LABEL:UP.npz:DOWN.npz")
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--wlim", type=float, default=3.0)
    ap.add_argument("--dw", type=float, default=0.004)
    ap.add_argument("--kref", type=int, default=8)
    ap.add_argument("--kmax", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "dos.png"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = np.arange(-a.wlim, a.wlim + 0.5 * a.dw, a.dw)
    wide = np.arange(-12.0, a.wlim + 0.5 * a.dw, a.dw)   # per la densita' serve la coda a w < 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for n, spec in enumerate(a.sets):
        parts = spec.split(":")
        if len(parts) == 3:
            label, up, dn = parts
        else:
            # cartella di una run: ultima iterazione con le next_cubes
            run = parts[-1].rstrip("/")
            import glob
            its = sorted(glob.glob(os.path.join(run, "iter[0-9][0-9][0-9]", "next_cubes", "A_komega_spinup_iter*.npz")))
            if not its:
                print(f"{run}: nessuna next_cubes, salto")
                continue
            up = its[-1]
            dn = up.replace("spinup", "spindown")
            label = parts[0] if len(parts) == 2 else f"{os.path.basename(run)} it{int(up[-7:-4])}"
        col = CAT[n % len(CAT)]
        for r, (spin, path) in enumerate((("up", up), ("down", dn))):
            N, mu, mass = dos(path, w, a.gamma, a.kref, a.kmax)
            Nw, _, _ = dos(path, wide, a.gamma, a.kref, a.kmax)
            dens = np.trapezoid(Nw[wide < 0], wide[wide < 0])
            i0 = int(np.argmin(np.abs(w)))
            print(f"{label:14s} {spin:4s}: N(0) = {N[i0]:.3f} ({N[i0] / (mass / 2):.2f} del libero)  "
                  f"n = int_(w<0) N = {dens:.4f}  (mu/2 = {mu / 2:.4f})", flush=True)
            for c, lim in ((0, a.wlim), (1, 0.4)):
                m = np.abs(w) <= lim
                axs[r, c].plot(w[m], N[m], lw=1.2, color=col, label=label)
        for r, spin in enumerate(("↑", "↓")):
            for c in (0, 1):
                axs[r, c].axhline(0.5, color="#6b6b68", lw=0.7, ls=":")
                axs[r, c].axvline(0, color="#6b6b68", lw=0.7)
                axs[r, c].set(xlabel="ω", ylabel=f"N_{spin}(ω)")
            axs[r, 0].set_title(f"N_{spin}(ω), Γ = {a.gamma:g}")
            axs[r, 1].set_title(f"N_{spin}(ω) vicino a ω = 0")
    axs[0, 0].legend(fontsize=8, frameon=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
