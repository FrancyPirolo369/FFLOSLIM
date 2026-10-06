#!/usr/bin/env python3
"""Confronto fetta per fetta: ImPi del prototipo (eb_proto, prodotto completo con quadrature adattate)
contro il metodo di produzione (residuo a griglia A*A - A0*A0 + parte coerente QPxQP analitica).

Il metodo di produzione e' preso nella versione MIGLIORE che sappiamo fare: residuo con l'integrando di
impi_table (test/p_integrand_probe.py) integrato in p con Gauss-Legendre a pannelli (GL16, come
test/p_gl_probe.py) e parte coerente di fflo.impi_cv con i parametri di produzione (nk 96, 48 nodi
angolari) e ad alta risoluzione (nk 384, 480).  Se le definizioni coincidono, la differenza residua fra
i due metodi e' errore di integrazione.  Questo script legge la produzione, non la modifica.

Uso (dalla radice di SLIM):  python3 test/energy_bubble/eb_compare.py [RUN]
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
sys.path.insert(0, os.path.join(HERE, "test", "energy_bubble"))
import eb_proto as E  # noqa: E402
import impi_union_probe as U  # noqa: E402
from p_gl_probe import breakpoints, gl_rule, f_at  # noqa: E402
from fflo.qp_cube import qp_model_from_cube  # noqa: E402
from fflo.impi_cv import _cv_init, _cv_work, _canonical_qp_fields, _kf  # noqa: E402

CASES = [(1.0, 0.3), (1.0, -0.05), (0.014, 0.3), (0.0, 0.3), (1.0, 1.0), (0.5, -0.3)]


def main(argv):
    run = argv[0] if argv else "P0p75_gmax"
    up = sorted(glob.glob(os.path.join(HERE, "out", "cluster", run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    work = os.path.join(HERE, "out", "impi_probe", "highP_integration", run)
    qff = U.setup(up, dn, os.path.join(work, "A0_up.npz"), os.path.join(work, "A0_dn.npz"),
                  n_p=31, dw=2.5e-4, n_linear=160)
    kfu, kfd = U.G["kf"]
    mu_, md_ = qp_model_from_cube(up), qp_model_from_cube(dn)
    ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, mu_)
    kd, ed, zd, gd, mu_dn = _canonical_qp_fields(dn, md_)
    feats = [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)]
    fup, fdn = E.Field.from_cube(up, "up"), E.Field.from_cube(dn, "down")
    print(f"{run}: qff = {qff:.4f}.  ImPi (x 1e3).  vecchio = residuo GL16 + coerente;  nuovo = eb_proto (24, 8)")
    print(f"{'fetta':20s} {'residuo':>9s} {'coer.prod':>10s} {'coer.alta':>10s} {'vecchio':>9s} {'nuovo':>9s} {'diff':>8s}")
    for qr, om in CASES:
        qv = qr * qff
        bp = breakpoints(qv, kfu, kfd, 0.15)
        x, wt = gl_rule(bp, 16)
        res = float(np.sum(wt * f_at(x, qv, om))) / (4 * math.pi)
        coh = []
        for nk, nq in ((96, 48), (384, 480)):
            _cv_init(np.array([om]), ku, eu, zu, gu, kd, ed, zd, gd, feats,
                     dict(angle_mode="kjac", nk=nk, nphi=64, n_kquad=nq, kquad_chunk_size=8, kmax=4.0, gamma_floor=1.0e-3))
            coh.append(float(_cv_work(qv)[0]))
        t0 = time.time()
        new = E.impi(qv, om, fup, fdn, n_pk=24, n_reg=8, n_th_reg=8, n_eps=6)
        old = res + coh[1]
        print(f"Q={qr:5.3f}qff Om={om:+5.2f} {1e3 * res:+9.4f} {1e3 * coh[0]:+10.4f} {1e3 * coh[1]:+10.4f} "
              f"{1e3 * old:+9.4f} {1e3 * new:+9.4f} {100 * (new / old - 1):+7.2f}%   [{time.time() - t0:.0f} s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
