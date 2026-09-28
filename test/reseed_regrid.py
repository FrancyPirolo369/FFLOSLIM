#!/usr/bin/env python3
"""Continuazione in polarizzazione con la griglia k rifatta per il P nuovo.

reseed_mu.py cambia solo mu: la griglia k resta quella del P di partenza, con i
pacchetti di nodi (larghi 0.02) sui kF VECCHI, e density la eredita per sempre (le cube
di uscita copiano k dal seed).  Un passo di 0.05 in P sposta kF_dn di 0.05-0.07, fuori
dal pacchetto.  Qui invece:
  1. Sigma della soluzione di partenza portata sulla griglia omega di base, come fa
     density prima del mixing (fflo.sigma_engine.load_cube_sigma);
  2. griglia k nuova: sotto k = 2 pacchetti sui kF nuovi (make_free_seed.k_grid), sopra
     k = 2 i nodi vecchi; Sigma interpolata linearmente in k (sopra k = 2 i nodi
     coincidono e Sigma e' copiata);
  3. A ricostruita con i mu nuovi dalla stessa fflo.density.pole_aware_rebuild che
     density usa per le cube di uscita: sigma0 in modo Fermi (= ReSigma(kF nuovo, 0)),
     righe omega locali attorno ai poli, satellite di contatto se la cube lo porta.
Il resto della cube (chiavi, convenzioni) e' copiato dalla sorgente.  Stessi nomi di
uscita di reseed_mu.py (A_komega_spin{up,down}_reseed.npz).

Uso (dalla radice):
  python3 test/reseed_regrid.py --up SRC_UP.npz --down SRC_DOWN.npz --P 0.75 --out-dir DIR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))


def scalar(z, key, default=np.nan):
    return float(np.asarray(z[key]).reshape(-1)[0]) if key in z else default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", required=True)
    ap.add_argument("--down", required=True)
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--mu-avg", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pole-refind-n-local", type=int, default=61)   # quello di run_test
    a = ap.parse_args()

    from scipy.interpolate import interp1d
    from fflo.density import occupied_nk, pole_aware_rebuild
    from fflo.sigma_engine import load_cube_sigma
    from make_free_seed import k_grid

    mu = {"up": a.mu_avg * (1.0 + a.P), "down": a.mu_avg * (1.0 - a.P)}
    kf = {s: np.sqrt(mu[s]) for s in mu}
    src = {"up": a.up, "down": a.down}
    cubes = {s: dict(np.load(p, allow_pickle=True)) for s, p in src.items()}
    k_old = np.asarray(cubes["up"]["k"], float)
    if not np.array_equal(k_old, np.asarray(cubes["down"]["k"], float)):
        raise SystemExit("le due cube non hanno la stessa griglia k")
    k_new = k_grid(k_old, kf["down"], kf["up"])
    outer = k_old > 2.0
    assert np.array_equal(k_new[outer], k_old[outer])

    def near(k, c):
        return int(np.sum(np.abs(k - c) < 0.02))

    old_mu = (scalar(cubes["up"], "mu_up"), scalar(cubes["up"], "mu_dn", scalar(cubes["up"], "mu_down")))
    print(f"P = {a.P:g}: mu {old_mu[0]:.4f}/{old_mu[1]:.4f} -> {mu['up']:.4f}/{mu['down']:.4f}, "
          f"kF {kf['up']:.4f}/{kf['down']:.4f}")
    print(f"nodi k entro 0.02 da kF_dn nuovo: griglia vecchia {near(k_old, kf['down'])}, "
          f"nuova {near(k_new, kf['down'])};  da kF_up: {near(k_old, kf['up'])} -> {near(k_new, kf['up'])}")

    os.makedirs(a.out_dir, exist_ok=True)
    for spin in ("up", "down"):
        z = cubes[spin]
        other = "down" if spin == "up" else "up"
        re_b, im_b = load_cube_sigma(Path(src[spin]))
        w_base = np.asarray(z["w_base"], float) if "w_base" in z else np.asarray(z["w"], float)
        re_n = interp1d(k_old, re_b, axis=0, assume_sorted=True)(k_new)
        im_n = np.minimum(interp1d(k_old, im_b, axis=0, assume_sorted=True)(k_new), 0.0)
        contact = str(np.asarray(z.get("high_k_sigma_mode", "stale")).reshape(-1)[0]) == "pair-contact"
        A, w, re_rows, im_rows, sigma0 = pole_aware_rebuild(
            k=k_new, w_base=w_base, re_sigma=re_n, im_sigma=im_n, mu=mu[spin],
            mass=scalar(z, "mass"), eta=scalar(z, "eta"),
            convention=str(np.asarray(z["a_convention"]).reshape(-1)[0]),
            occupied_protect_min=-102.0, sigma0_mode="fermi", sigma0_density_kmax=np.nan,
            pole_refind_n_local=a.pole_refind_n_local,
            high_k_contact_start=scalar(z, "high_k_contact_start") if contact else np.nan,
            high_k_contact_delta2=scalar(z, "pair_delta_inf_squared") if contact else np.nan,
            high_k_contact_mu_internal=mu[other])
        n = occupied_nk(A, w)
        dens = np.trapezoid(k_new * n, k_new)
        z.update(k=k_new, A=A, w=w, w_base=w_base, ReS=re_rows, ImS=im_rows,
                 sigma0=np.array(float(sigma0)), mu_up=np.array(mu["up"]),
                 mu_dn=np.array(mu["down"]), mu_down=np.array(mu["down"]),
                 reseed_from=np.array(os.path.abspath(src[spin])),
                 reseed_mode=np.array("regrid"))
        out = os.path.join(a.out_dir, f"A_komega_spin{spin}_reseed.npz")
        np.savez(out, **z)
        print(f"{spin:4s}: sigma0 {scalar(cubes[spin], 'sigma0'):+.4f} -> {float(sigma0):+.4f}   "
              f"densita' della cube {dens:.4f} = mu/2 {dens / (mu[spin] / 2) - 1:+.2%}   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
