#!/usr/bin/env python3
"""Copia una coppia di cube cambiando solo mu (continuazione in polarizzazione).

Il driver legge mu_up / mu_dn dalle cube di partenza: per partire a P = 0.80 dalla
soluzione di P = 0.70 bisogna ricopiare le cube con i mu nuovi,
    mu_up = mu_avg (1 + P),  mu_dn = mu_avg (1 - P).
Sigma, sigma0, griglie e A restano quelle della soluzione di partenza: il primo passo di
density ricalcola Sigma con i mu nuovi (e sigma0 in modo Fermi).  La griglia in k resta
quella del P di partenza (pacchetti di nodi sui kF vecchi): accettabile per una
continuazione a passi piccoli, non per salti grandi in P.

Uso:  python3 test/reseed_mu.py --up SRC_UP.npz --down SRC_DOWN.npz --P 0.80 --out-dir DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", required=True)
    ap.add_argument("--down", required=True)
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--mu-avg", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    mu_up, mu_dn = a.mu_avg * (1.0 + a.P), a.mu_avg * (1.0 - a.P)
    os.makedirs(a.out_dir, exist_ok=True)
    for spin, src in (("up", a.up), ("down", a.down)):
        z = dict(np.load(src, allow_pickle=True))
        old = (float(np.asarray(z["mu_up"]).reshape(-1)[0]),
               float(np.asarray(z["mu_dn"]).reshape(-1)[0]))
        z["mu_up"] = np.array(mu_up)
        z["mu_dn"] = np.array(mu_dn)
        if "mu_down" in z:
            z["mu_down"] = np.array(mu_dn)
        z["reseed_from"] = np.array(os.path.abspath(src))
        out = os.path.join(a.out_dir, f"A_komega_spin{spin}_reseed.npz")
        np.savez(out, **z)
        print(f"{spin}: mu {old[0]:.4f}/{old[1]:.4f} -> {mu_up:.4f}/{mu_dn:.4f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
