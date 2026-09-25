#!/usr/bin/env python3
"""Seed di fermioni liberi per una polarizzazione non interagente P a mu medio fisso.

  mu_up = mu_avg (1 + P),  mu_dn = mu_avg (1 - P),  kF_s = sqrt(mu_s)   (xi = k^2/m, m = 1)
  P = (mu_up - mu_dn)/(mu_up + mu_dn): e' anche la P delle densita' se vale Luttinger
  (n_s = mu_s/2).  A P = 0.65 e mu_avg = 1 si ritrova mu = 1.65 / 0.35.

Cube: Sigma = 0 (ReS = ImS = 0, sigma0 = 0), A = lorentziana di larghezza eta.  Il
modello di quasiparticelle di pairbuild (Dyson su ReS/ImS) e la bolla (ricostruzione di A
da Sigma) sono quindi esatti anche sulla griglia omega unica.  Formato e griglia omega
presi dal seed warm di P = 0.65 (template); la griglia in k e' rifatta per questo P con
i pacchetti di nodi sui kF NUOVI (riusare la griglia di un altro P aveva dato eccessi di
densita' ~1%, nota ereditata).  Sopra k = K_SPLIT si tengono i nodi del template.

Uso (dalla radice del bundle/SLIM):
  python3 test/make_free_seed.py --P 0.40 --out-dir seeds_free/P0p40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
K_SPLIT = 2.0


def k_grid(template_k, kf_dn, kf_up, width=0.02, peak=12.0):
    sys.path.insert(0, os.path.join(HERE, "test"))
    from run_test import smooth_multicenter_grid   # la stessa di run_fflo, ma dentro test/
    outer = template_k[template_k > K_SPLIT]
    n_inner = template_k.size - outer.size
    centers = sorted({round(kf_dn, 12), round(kf_up, 12)})
    inner = smooth_multicenter_grid(float(template_k[0]), K_SPLIT, n_inner + 1, centers,
                                    [width] * len(centers), [peak] * len(centers))[:-1]
    k = np.concatenate([inner, outer])
    assert np.all(np.diff(k) > 0) and k.size == template_k.size
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--mu-avg", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--template-dir", default=os.path.join(HERE, "seeds"))
    a = ap.parse_args()
    if not 0.0 <= a.P < 1.0:
        ap.error("serve 0 <= P < 1")
    mu = {"up": a.mu_avg * (1.0 + a.P), "down": a.mu_avg * (1.0 - a.P)}
    tu = np.load(os.path.join(a.template_dir, "A_komega_spinup_warm.npz"), allow_pickle=True)
    k = k_grid(np.asarray(tu["k"], float), np.sqrt(mu["down"]), np.sqrt(mu["up"]))
    w = np.asarray(tu["w"], float)
    if w.ndim != 1:
        raise SystemExit("il template deve avere una griglia omega unica (w 1D)")
    wb = np.asarray(tu["w_base"], float) if "w_base" in tu.files else w
    eta = float(np.asarray(tu["eta"]).reshape(-1)[0])
    os.makedirs(a.out_dir, exist_ok=True)
    for spin in ("up", "down"):
        xi = k[:, None] ** 2 - mu[spin]
        A = (eta / np.pi) / ((w[None, :] - xi) ** 2 + eta ** 2)
        zeros = np.zeros_like(A)
        out = os.path.join(a.out_dir, f"A_komega_spin{spin}_free.npz")
        np.savez(out, k=k, w=w, w_base=wb, A=A, ReS=zeros, ImS=zeros,
                 sigma0=np.array(0.0), sigma0_w=np.array(0.0), mass=np.array(1.0),
                 eta=np.array(eta), a_convention=tu["a_convention"],
                 mu_up=np.array(mu["up"]), mu_dn=np.array(mu["down"]),
                 mu_down=np.array(mu["down"]), spin=np.array(spin),
                 mode=np.array("free_seed"), model=np.array("free_seed"),
                 eps0=np.array(1.0), epsilon_0=np.array(1.0), pmf_scale=np.array(0.0),
                 iteration=np.array(0))
    print(f"P={a.P:g}: mu_up={mu['up']:.4f} mu_dn={mu['down']:.4f} "
          f"kF={np.sqrt(mu['up']):.4f}/{np.sqrt(mu['down']):.4f} "
          f"qff={np.sqrt(mu['up']) - np.sqrt(mu['down']):.4f}  -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
