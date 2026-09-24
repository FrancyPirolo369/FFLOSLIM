#!/usr/bin/env python3
"""Sigma e bilanci di density per iterazione, da rsyncare invece delle cube.

Per ogni out/<RUN>/iterNNN completa scrive out/<RUN>/snap/sigmaNNN.npz con, per
spin s in (up, dn):
  rows_k_s, rows_w_s, rows_ReS_s, rows_ImS_s, rows_A_s   righe della cube emessa
                                   a k = 0, 0.3, 0.5, kF_dn, 0.8, 1, kF_up, 1.6, 2, 3, 4, 6, 8
  k_s, reS0_s                      ReSigma(k, w=0) per tutti i k
  sumA_s                           int A(k, w) dw per riga (regola di somma)
  sigma0_s
  dr_*                             tutto density_results.npz (n(k) del seed, fresca
                                   alpha=1 e mescolata, somme, sigma0, densita')
  ck_s_*                           checkpoint di ImSigma grezza ai nodi (k_sparse x omega)
Circa 0.5 MB per iterazione.  Le iterazioni gia' estratte sono saltate.

Uso (dalla radice del bundle/SLIM):  python3 test/extract_sigma.py RUN [RUN ...]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cube_rows(path, targets):
    z = np.load(path, allow_pickle=True)
    k = np.asarray(z["k"], dtype=float)
    w = np.asarray(z["w"], dtype=float)
    res, ims, a = (np.asarray(z[key], dtype=float) for key in ("ReS", "ImS", "A"))
    if w.ndim == 1:
        w = np.broadcast_to(w, res.shape)
    idx = np.unique([int(np.argmin(np.abs(k - t))) for t in targets])
    re0 = np.array([np.interp(0.0, w[j], res[j]) for j in range(k.size)])
    suma = np.array([np.trapezoid(a[j], w[j]) for j in range(k.size)])
    return dict(rows_k=k[idx], rows_w=w[idx].astype(np.float32),
                rows_ReS=res[idx].astype(np.float32), rows_ImS=ims[idx].astype(np.float32),
                rows_A=a[idx].astype(np.float32), k=k, reS0=re0, sumA=suma,
                sigma0=float(np.asarray(z["sigma0"]).reshape(-1)[0]),
                mu_up=float(np.asarray(z["mu_up"]).reshape(-1)[0]),
                mu_dn=float(np.asarray(z["mu_dn"]).reshape(-1)[0]))


def extract(run_dir):
    snap = os.path.join(run_dir, "snap")
    os.makedirs(snap, exist_ok=True)
    done = 0
    for itd in sorted(glob.glob(os.path.join(run_dir, "iter[0-9][0-9][0-9]"))):
        dst = os.path.join(snap, "sigma" + os.path.basename(itd)[4:] + ".npz")
        if os.path.exists(dst):
            continue
        up = sorted(glob.glob(os.path.join(itd, "next_cubes", "*spinup*.npz")))
        dn = sorted(glob.glob(os.path.join(itd, "next_cubes", "*spindown*.npz")))
        dres = os.path.join(itd, "density", "density_results.npz")
        if not (up and dn and os.path.exists(dres)):
            continue
        out = {}
        first = None
        for s, path in (("up", up[0]), ("dn", dn[0])):
            if first is None:
                zz = np.load(path, allow_pickle=True)
                mu_up = float(np.asarray(zz["mu_up"]).reshape(-1)[0])
                mu_dn = float(np.asarray(zz["mu_dn"]).reshape(-1)[0])
                first = [0.0, 0.3, 0.5, np.sqrt(mu_dn), 0.8, 1.0, np.sqrt(mu_up),
                         1.6, 2.0, 3.0, 4.0, 6.0, 8.0]
            for key, val in cube_rows(path, first).items():
                out[f"{key}_{s}"] = val
        with np.load(dres, allow_pickle=True) as z:
            for key in z.files:
                out["dr_" + key] = np.asarray(z[key])
        for s, name in (("up", "up"), ("dn", "down")):
            ck = os.path.join(itd, "density", f"imsigma_{name}_checkpoint.npz")
            if os.path.exists(ck):
                with np.load(ck) as z:
                    for key in ("k_sparse", "omega", "im_sparse"):
                        out[f"ck_{s}_{key}"] = np.asarray(z[key])
        np.savez_compressed(dst, **out)
        done += 1
    return done


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    for r in argv:
        d = os.path.join(ROOT, "out", r)
        if os.path.isdir(d):
            print(f"{r}: {extract(d)} nuove iterazioni estratte")
        else:
            print(f"{r}: cartella {d} assente")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
