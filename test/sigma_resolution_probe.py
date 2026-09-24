#!/usr/bin/env python3
"""Da dove vengono i wiggles di n(k) k^4 fra k=3 e 8?  Misura diretta su poche righe.

In produzione (fflo.density.memory_safe_im_sigma) ImSigma e' calcolata su
sigma_nk nodi in k e sigma_nomega nodi in omega, poi interpolata: in k a omega
fisso (lineare), in omega con Pchip.  Con sigma_nomega=41 sotto omega=-4 ci
sono solo i nodi -6.8, -12, -21, -38, -69, -122, -216, mentre il satellite di
contatto sta a omega ~ mu_int - k^2 (da -14 a -62 per k=4..8).

Questa sonda calcola ImSigma ESATTA (ogni riga e' un nodo) su righe k scelte e
su una griglia omega fitta sul satellite unita ai nodi di produzione, e
confronta la n(k) k^4 di Tan
    n(k) ~ int_{w<0} (-ImSigma/pi) / (w - xi(k))^2,   xi = k^2/m - (mu + sigma0)
in tre versioni:
    fitta   tutte le colonne omega (la verita', a questa risoluzione)
    prod    solo i sigma_nomega nodi di produzione + Pchip in omega
    kint    righe dispari ricostruite interpolando in k le pari (passo doppio)
Se prod oscilla e fitta no, la manopola e' sigma_nomega, non sigma_nk.

Uso (dalla radice):
    python3 test/sigma_resolution_probe.py --pair-table P.npz --seed-up U.npz \
        --seed-down D.npz --out DIR [--pintmax 12 --k 4,4.5,...]
Per il cluster: --pair-table iterNNN/pairbuild/pair/pair_gamma_table.npz e le
cube di ingresso di quell'iterazione.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-table", required=True)
    ap.add_argument("--seed-up", required=True)
    ap.add_argument("--seed-down", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spin", default="down", choices=("up", "down"))
    ap.add_argument("--pintmax", type=float, default=12.0)
    ap.add_argument("--k", default="4,4.5,5,5.5,6,6.5,7,7.5")
    ap.add_argument("--n-dense", type=int, default=100)
    ap.add_argument("--dense-min", type=float, default=-90.0)
    ap.add_argument("--dense-max", type=float, default=-2.0)
    ap.add_argument("--sigma-nomega", type=int, default=41)
    ap.add_argument("--n-theta", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delta", type=float, default=1.0e-3)
    a = ap.parse_args()

    # stesse variabili che test/run_test.py passa a density (lette a runtime)
    os.environ["SIGMA_PN_PINTMAX"] = f"{a.pintmax:g}"
    os.environ.setdefault("SIGMA_PN_RING_MODE", "exact_window")
    os.environ.setdefault("SIGMA_PN_ANGLE_EXACT_CUT", "1")
    os.environ.setdefault("SIGMA_PN_Q_GL_N", "16")
    sys.path.insert(0, HERE)
    from scipy.interpolate import PchipInterpolator
    from fflo.density import load_seed, memory_safe_im_sigma
    from fflo.pair_shifted import apply_shift_to_pair_table, load_pair_table
    from fflo.sigma_engine import Physics

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = {"up": load_seed(Path(a.seed_up)), "down": load_seed(Path(a.seed_down))}
    mu_up = float(seeds["up"]["mu_up"])
    mu_dn = float(seeds["up"]["mu_dn"])
    qff = float(np.sqrt(mu_up) - np.sqrt(mu_dn))
    physics = Physics(eps0=1.0, mu_up=mu_up, mu_down=mu_dn, mass=1.0, pmf_scale=0.0)

    # pair come in density: shift a qff, delta, floor broad
    pair = load_pair_table(a.pair_table)
    q_t, w_t = np.asarray(pair["q"]), np.asarray(pair["omega"])
    iq = int(np.argmin(np.abs(q_t - qff)))
    shifted = apply_shift_to_pair_table(pair, float(q_t[iq]), subcritical_delta=a.delta,
                                        eta_floor_mode="broad")

    # nodi omega di produzione: stessa costruzione di density.py
    w_base = np.asarray(seeds["up"]["w"], dtype=float)
    if w_base.ndim > 1:
        w_base = np.asarray(seeds["up"]["w_base"], dtype=float)
    wu = w_base[(w_base >= -216.0 - 1e-12) & (w_base <= 144.0 + 1e-12)]
    n = min(a.sigma_nomega, wu.size)
    w_prod = wu[np.unique(np.round(np.linspace(0, wu.size - 1, n)).astype(int))]
    w_dense = np.linspace(a.dense_min, a.dense_max, a.n_dense)
    w_all = np.unique(np.concatenate([w_prod, w_dense]))
    k_rows = np.array([float(v) for v in a.k.split(",")])

    ext = a.spin
    cube_int = seeds["down" if ext == "up" else "up"]
    print(f"spin {ext}: {k_rows.size} righe k x {w_all.size} omega "
          f"({w_prod.size} di produzione), PINTMAX {a.pintmax:g}", flush=True)
    im = memory_safe_im_sigma(
        spin=ext, cube=cube_int, pair_q=np.asarray(shifted["q"]),
        pair_omega=np.asarray(shifted["omega"]), pair_a=np.asarray(shifted["A_pair"]),
        k_target=k_rows, omega_target=w_all, physics=physics, qff=float(q_t[iq]),
        sigma_nk=k_rows.size, sigma_k_feature_fraction=0.0, n_theta=a.n_theta,
        omega_chunk=16, workers=a.workers, checkpoint=out / f"probe_{ext}_checkpoint.npz")
    im = np.minimum(np.asarray(im, dtype=float), 0.0)

    mu_ext = mu_up if ext == "up" else mu_dn
    sigma0 = float(seeds[ext]["sigma0"])
    in_prod = np.isin(w_all, w_prod)

    def tan(k, w_nodes, im_row):
        """k^4 int_{w<0} (-ImS/pi)/(w - xi)^2 con ImS Pchip sui nodi dati."""
        xi = k * k - (mu_ext + sigma0)
        lo = max(w_nodes[0], -(k * k) - 60.0)
        wf = np.linspace(lo, 0.0, 40001)
        f = np.minimum(PchipInterpolator(w_nodes, im_row, extrapolate=False)(wf), 0.0)
        f = np.nan_to_num(f)
        return k ** 4 * np.trapezoid(-f / np.pi / (wf - xi) ** 2, wf)

    dense = np.array([tan(k, w_all, im[i]) for i, k in enumerate(k_rows)])
    prod = np.array([tan(k, w_prod, im[i, in_prod]) for i, k in enumerate(k_rows)])
    kint = np.full_like(dense, np.nan)
    for i in range(1, k_rows.size - 1, 2):
        t = (k_rows[i] - k_rows[i - 1]) / (k_rows[i + 1] - k_rows[i - 1])
        row = (1 - t) * im[i - 1] + t * im[i + 1]
        kint[i] = tan(k_rows[i], w_all, row)
    np.savez_compressed(out / f"probe_{ext}.npz", k=k_rows, omega=w_all, im=im,
                        w_prod=w_prod, nk4_dense=dense, nk4_prod=prod, nk4_kint=kint,
                        sigma0=sigma0, mu_ext=mu_ext)
    print(f"\n   k    fitta   prod(41)  prod/fitta   kint(passo {2 * np.diff(k_rows).mean():.2g})")
    for i, k in enumerate(k_rows):
        ki = f"{kint[i]:.4f} ({kint[i] / dense[i] - 1:+.1%})" if np.isfinite(kint[i]) else ""
        print(f"{k:5.2f}  {dense[i]:.4f}  {prod[i]:.4f}    {prod[i] / dense[i] - 1:+6.1%}    {ki}")
    print(f"\nscarto (max-min)/2 / media:  fitta {np.ptp(dense) / 2 / dense.mean():.1%}   "
          f"prod {np.ptp(prod) / 2 / prod.mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
