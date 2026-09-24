#!/usr/bin/env python3
"""Rimette nella tabella ImPi il peso che la bolla numerica perde fuori dal disco.

IL PROBLEMA (misurato il 2026-09-24)
La bolla numerica (fflo.impi_table) integra l'impulso ASSOLUTO del fermione up,
|k_up| < Lambda (p_int_max, 4 in produzione), con il down a |Q - k_up|.  Il
riferimento contro cui pair_gamma fa il KK del residuo e' invece la Gamma_0
analitica senza cutoff (vuoto galileiano + mezzo libero, cioe' la Gamma_0 di
Pirolo-Pisani-Pieri 2026).  Il residuo ImPi_num - ImPi_ref contiene quindi, oltre
alla fisica, tutto il peso libero fuori dal disco: a Q >= 10 e' l'intero 1/8 del
vuoto, e il suo KK sposta lo stato legato fino alla soglia.

LA CORREZIONE (sottrazione nell'integrando, alla Pini/Enss)
Il riferimento va troncato sullo STESSO disco:
    residuo = ImPi_num[disco] - ImPi_0[disco]
e poi si somma ImPi_0 completo.  Equivalentemente, senza toccare pair_gamma:
    ImPi_fixed = ImPi_num + ImPi_0[fuori dal disco]
La parte fuori dal disco e' calcolata per fermioni liberi, T=0, eps_k = k^2/m:
con k_up = Q/2 + p, k_down = Q/2 - p l'energia e' Q^2/(2m) + 2p^2/m, quindi la
delta fissa |p| = r e resta un integrale angolare sull'anello:
    ImPi_0(Q, W) = -(m/8) <[1 - th(-xi_up) - th(-xi_dn)]>_phi
(normalizzazione verificata contro fflo.kramers_kronig.proxy_full_pi_from_gamma:
vedi --check).  L'errore residuo e' il vestimento (Sigma) fuori dal disco, che
decade con k; deve rendere Gamma indipendente da Lambda: e' il test.

Uso:
    python3 test/fix_truncation.py --check                  # normalizzazione
    python3 test/fix_truncation.py IN_impi_table.npz OUT.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ring_average(q, omega, mu_up, mu_dn, mass=1.0, lam=None, outside=True,
                 n_phi=4096, shift_up=0.0, shift_dn=0.0):
    """-(m/8) * angular average over the relative-momentum ring of the free
    Pauli factor, optionally restricted to |k_up| > lam (outside=True) or
    |k_up| <= lam (outside=False).  Returns an array shaped like omega.

    shift_s moves the dispersion, E_s(k) = k^2/m - mu_s + shift_s, while the
    Pauli factor keeps the Fermi surfaces at k_Fs = sqrt(m mu_s) (Luttinger).
    With shift_s = -sigma0_s the energy is k^2/m - (mu_s + sigma0_s), i.e. the
    physical chemical potential that the large-k fermions of the cube see."""
    w = np.asarray(omega, dtype=float)
    q = float(q)
    # 2 p^2/m = W + mu_up + mu_dn - Q^2/(2m)
    r2 = 0.5 * mass * (w + mu_up + mu_dn - shift_up - shift_dn
                       - q * q / (2.0 * mass))
    out = np.zeros_like(w)
    ok = r2 > 0.0
    if not np.any(ok):
        return out
    r = np.sqrt(r2[ok])[:, None]
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    c = np.cos(phi)[None, :]
    kup2 = r * r + 0.25 * q * q + r * q * c
    kdn2 = r * r + 0.25 * q * q - r * q * c
    pauli = (1.0 - (kup2 / mass < mu_up).astype(float)
             - (kdn2 / mass < mu_dn).astype(float))
    if lam is not None:
        sel = kup2 > lam * lam if outside else kup2 <= lam * lam
        pauli = pauli * sel
    out[ok] = -(mass / 8.0) * pauli.mean(axis=1)
    return out


def check():
    """The full ring (no disk) must reproduce Im of the code's analytic proxy."""
    from fflo.kramers_kronig import proxy_full_pi_from_gamma
    mu_up, mu_dn, eps0 = 1.65, 0.35, 1.0
    omega = np.linspace(-6.0, 40.0, 2301)
    worst = 0.0
    for q in (0.0, 0.3, 0.6929, 1.0, 2.0, 4.0, 8.0):
        mine = ring_average(q, omega, mu_up, mu_dn, n_phi=20000)
        ref = np.imag(proxy_full_pi_from_gamma(
            omega, q=q, eps0=eps0, mu_up=mu_up, mu_dn=mu_dn,
            eta_gamma=1e-6, cutoff=4.0))
        # skip the cells right at kinks, where eta and the phi grid differ
        err = np.abs(mine - ref)
        med = float(np.median(err))
        p99 = float(np.quantile(err, 0.99))
        worst = max(worst, med)
        print(f"Q={q:6.4f}  median|diff|={med:.2e}  p99|diff|={p99:.2e}  "
              f"max|ref|={np.abs(ref).max():.4f}")
    print("CHECK", "OK" if worst < 1e-3 else "FAILED", f"(worst median {worst:.2e})")
    return worst < 1e-3


def cube_sigma0(path):
    z = np.load(path, allow_pickle=True)
    return float(np.asarray(z["sigma0"]).reshape(-1)[0])


def fix_table(src, dst, n_phi=4096, mu_mode="physical"):
    z = dict(np.load(src, allow_pickle=True))
    q = np.asarray(z["q"], dtype=float)
    w = np.asarray(z["omega"], dtype=float)
    lam = float(np.asarray(z["p_int_max"]).reshape(-1)[0])
    mu_up = float(np.asarray(z["mu_up"]).reshape(-1)[0])
    mu_dn = float(np.asarray(z["mu_dn"]).reshape(-1)[0])
    s_up = s_dn = 0.0
    if mu_mode == "physical":
        # outside the disk the fermions are at k > Lambda: there Re Sigma -> 0
        # and G = 1/(w - k^2/m + mu + sigma0), so the energy carries -sigma0
        s_up = -cube_sigma0(str(np.asarray(z["up_cube_path"])))
        s_dn = -cube_sigma0(str(np.asarray(z["down_cube_path"])))
    missing = np.array([ring_average(qi, w, mu_up, mu_dn, lam=lam, outside=True,
                                     n_phi=n_phi, shift_up=s_up, shift_dn=s_dn)
                        for qi in q])
    z["ImPiDisk"] = np.asarray(z["ImPi"], dtype=float).copy()
    z["ImPiOutsideDiskFree"] = missing
    for key in ("ImPi", "ImPiFull"):
        if key in z:
            z[key] = np.asarray(z[key], dtype=float) + missing
    z["ImPi_truncation_fix"] = np.array(
        f"free ring outside |k_up|<p_int_max added, mu_mode={mu_mode}, "
        f"energy shifts up={s_up:+.6f} dn={s_dn:+.6f}")
    np.savez_compressed(dst, **z)
    i10 = int(np.argmin(np.abs(q - 10.0)))
    print(f"Lambda={lam:g} mu_mode={mu_mode} shifts {s_up:+.3f}/{s_dn:+.3f}  "
          f"nQ={q.size} nW={w.size}  "
          f"max|added|={np.abs(missing).max():.4f}  "
          f"max|added| at Q={q[i10]:.2f}: {np.abs(missing[i10]).max():.4f}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n-phi", type=int, default=4096)
    ap.add_argument("--mu-mode", choices=("physical", "label"), default="physical",
                    help="energy of the fermions outside the disk: physical = "
                         "k^2/m - (mu + sigma0) (what the cube has at large k), "
                         "label = k^2/m - mu (the analytic reference)")
    a = ap.parse_args()
    if a.check:
        return 0 if check() else 1
    if not (a.src and a.dst):
        ap.error("need SRC and DST (or --check)")
    fix_table(a.src, a.dst, a.n_phi, a.mu_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
