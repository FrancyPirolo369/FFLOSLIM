#!/usr/bin/env python3
"""Correzione a un passo di g_c da una bolla integrata bene, P per P (sola riga Q = qff).

Per ogni run, sulle cube dell'ultima iterazione:
    ImPi_vecchio(qff, Omega) = residuo a griglia come in produzione (impi_table: reticolo dw 1e-3 con
                               coda 100, 31 nodi in p su [0, 4], phipanel 10 + 7) + parte coerente
                               QPxQP (fflo.impi_cv, nk 96, 48 nodi angolari, gamma_floor 1e-3)
    ImPi_nuovo(qff, Omega)   = eb_proto.impi (prodotto completo, quadrature adattate)
    DeltaReGamma^-1(qff, 0)  = (1/pi) int dOmega [-(ImPi_nuovo - ImPi_vecchio)] / Omega   (|Omega| <= 12)
    Delta g_c (un passo)     = -4 pi DeltaReGamma^-1(qff, 0)
Il punto fisso amplifica (a P = 0.50 circa x5): qui conta la FORMA in P, da confrontare con gli
scarti dalla spline della linea critica.  La produzione e' solo letta.

Uso (dalla radice di SLIM):  python3 test/energy_bubble/eb_qff_scan.py [RUN ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
sys.path.insert(0, os.path.join(HERE, "test", "energy_bubble"))
import eb_proto as E  # noqa: E402

RUNS = ["P0p10_prod", "P0p20_prod", "P0p30_prod", "P0p40_prod", "P0p50_prod", "P0p60_prod",
        "P0p65_prod", "P0p70_gmax"]
_F = {}


def omega_grid():
    pos = np.geomspace(5e-4, 12.0, 50)
    return np.concatenate((-pos[::-1], pos))


def _new_point(task):
    qv, om = task
    return E.impi(qv, om, _F["up"], _F["dn"], n_pk=16, n_reg=6, n_th_reg=6, n_eps=4)


def one_run(run, workers):
    base = os.path.join(HERE, "out", "cluster", run)
    up = sorted(glob.glob(os.path.join(base, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    work = os.path.join(HERE, "out", "energy_bubble", "qff_scan", run)
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "qff_row.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    import impi_union_probe as U
    from fflo.qp_cube import build_qp_cube, qp_model_from_cube
    from fflo.impi_cv import _cv_init, _cv_work, _canonical_qp_fields, _kf
    om = omega_grid()
    mu_, md_ = qp_model_from_cube(up), qp_model_from_cube(dn)
    a0u, a0d = os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz")
    if not os.path.exists(a0u):
        build_qp_cube(up, a0u, model=mu_)
        build_qp_cube(dn, a0d, model=md_)
    t0 = time.time()
    qff = U.setup(up, dn, a0u, a0d, n_p=31, dw=1e-3, n_linear=40)
    res = U.run([qff], om, "lattice", workers)[0]
    ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, mu_)
    kd, ed, zd, gd, mu_dn = _canonical_qp_fields(dn, md_)
    _cv_init(om, ku, eu, zu, gu, kd, ed, zd, gd, [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)],
             dict(angle_mode="kjac", nk=96, nphi=64, n_kquad=48, kquad_chunk_size=8, kmax=4.0, gamma_floor=1.0e-3))
    coh = np.asarray(_cv_work(qff))
    t_old = time.time() - t0
    t0 = time.time()
    _F["up"], _F["dn"] = E.Field.from_cube(up, "up"), E.Field.from_cube(dn, "down")
    with mp.get_context("fork").Pool(workers) as pool:
        new = np.array(pool.map(_new_point, [(qff, float(o)) for o in om], chunksize=2))
    t_new = time.time() - t0
    out = dict(omega=om, old=res + coh, residual=res, coherent=coh, new=new, qff=qff,
               t_old=t_old, t_new=t_new, cube=os.path.basename(up))
    np.savez(path, **out)
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=RUNS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    print(f"{'run':14s} {'cubo':30s} {'dReG^-1(qff) [delta]':>21s} {'dg un passo':>12s}  tempi vecchio/nuovo")
    rows = []
    for run in a.runs:
        r = one_run(run, a.workers)
        om = np.asarray(r["omega"])
        d = float(np.trapezoid(-(np.asarray(r["new"]) - np.asarray(r["old"])) / om, om) / np.pi)
        dg = -4 * math.pi * d
        rows.append((run, dg))
        print(f"{run:14s} {str(r['cube']):30s} {d / 1e-3:+21.2f} {dg:+12.4f}  {float(r['t_old']):.0f}/{float(r['t_new']):.0f} s",
              flush=True)
    np.savez(os.path.join(HERE, "out", "energy_bubble", "qff_scan", "summary.npz"),
             runs=np.array([x for x, _ in rows]), dg=np.array([y for _, y in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
