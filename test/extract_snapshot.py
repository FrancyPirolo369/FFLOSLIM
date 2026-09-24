#!/usr/bin/env python3
"""Riassunto compatto di ogni iterazione, da rsyncare invece delle cube.

Per ogni out/<RUN>/iterNNN completa scrive out/<RUN>/snap/iterNNN.npz con:
  k, n_up(k), n_dn(k)                  dalle next_cubes (n = int_{w<=0} A dw)
  q, omega, ReInvGamma, ImInvGamma     dalla tabella di coppia di quell'iterazione
  pair_contact, logged_shift          da density_summary.txt (se lo shift e'
                                       congelato density scrive il valore IMPOSTO)
  reinv_qff_raw                        ReGamma^-1(qff, w~0) letto dalla tabella:
                                       distanza vera dalla criticita' anche a g fisso
  shift_used                           frozen_shift.txt se congelato, altrimenti = raw
Pochi MB per iterazione invece di ~20.  Le iterazioni gia' estratte sono saltate.

Uso (dalla radice del bundle/SLIM):  python3 test/extract_snapshot.py [RUN ...]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def occupation(cube):
    z = np.load(cube, allow_pickle=True)
    k, a, w = np.asarray(z["k"]), np.asarray(z["A"]), np.asarray(z["w"])
    if w.ndim == 1:
        m = w <= 0.0
        n = np.trapezoid(a[:, m], w[m], axis=1)
    else:
        n = np.array([np.trapezoid(a[j][w[j] <= 0.0], w[j][w[j] <= 0.0])
                      for j in range(k.size)])
    return k, n


def grab(text, key):
    for line in text.splitlines():
        if line.startswith(key + " "):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                return float("nan")
    return float("nan")


def frozen(run_dir):
    try:
        return float(open(os.path.join(run_dir, "frozen_shift.txt")).read().split()[0])
    except (OSError, ValueError, IndexError):
        return float("nan")


def extract(run_dir):
    snap = os.path.join(run_dir, "snap")
    os.makedirs(snap, exist_ok=True)
    done = 0
    for itd in sorted(glob.glob(os.path.join(run_dir, "iter[0-9][0-9][0-9]"))):
        tag = os.path.basename(itd)
        dst = os.path.join(snap, tag + ".npz")
        if os.path.exists(dst):
            continue
        up = sorted(glob.glob(os.path.join(itd, "next_cubes", "*spinup*.npz")))
        dn = sorted(glob.glob(os.path.join(itd, "next_cubes", "*spindown*.npz")))
        pair = os.path.join(itd, "pairbuild", "pair", "pair_gamma_table.npz")
        summ = os.path.join(itd, "density", "density_summary.txt")
        if not (up and dn and os.path.exists(pair) and os.path.exists(summ)):
            continue                      # iterazione non ancora completa
        k, n_up = occupation(up[0])
        k2, n_dn = occupation(dn[0])
        t = np.load(pair, allow_pickle=True)
        txt = open(summ).read()
        zc = np.load(up[0], allow_pickle=True)
        mu_up = float(np.asarray(zc["mu_up"]).reshape(-1)[0])
        mu_dn = float(np.asarray(zc["mu_dn" if "mu_dn" in zc.files else "mu_down"]).reshape(-1)[0])
        qff = float(np.sqrt(mu_up) - np.sqrt(mu_dn))
        q_t, w_t = np.asarray(t["q"]), np.asarray(t["omega"])
        raw = float(np.interp(qff, q_t, np.asarray(t["ReInvGamma"])[:, int(np.argmin(np.abs(w_t)))]))
        fz = frozen(run_dir)
        np.savez_compressed(
            dst, k=k, n_up=n_up, k_dn=k2, n_dn=n_dn,
            q=np.asarray(t["q"]), omega=np.asarray(t["omega"]),
            ReInvGamma=np.asarray(t["ReInvGamma"], dtype=np.float32),
            ImInvGamma=np.asarray(t["ImInvGamma"], dtype=np.float32),
            pair_contact=grab(txt, "pair_contact"),
            logged_shift=grab(txt, "pair_raw_thouless_shift"),
            reinv_qff_raw=raw, qff=qff,
            shift_used=fz if np.isfinite(fz) else raw)
        done += 1
    return done


def main(argv):
    runs = argv or [os.path.basename(os.path.dirname(p))
                    for p in glob.glob(os.path.join(ROOT, "out", "*", "loop.log"))]
    for r in sorted(runs):
        d = os.path.join(ROOT, "out", r)
        if os.path.isdir(d):
            print(f"{r}: {extract(d)} nuove iterazioni estratte")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
