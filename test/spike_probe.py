#!/usr/bin/env python3
"""Righe spurie di ReGamma^-1(Q,0) vicino a qff: quale pezzo di ImPi ha il picco a |Omega| ~ 0.01.

ImPi di pairbuild = residuo a griglia (A*A - A0*A0, impi_table) + QPxQP semi-analitico
(impi_cv).  Sulle stesse cube della tabella con le righe spurie:
  T1  i due pezzi separati, righe di Q della tabella attorno a qff, Omega fitto
  T2  Q fitto attorno a qff (fisica = continua in Q, quadratura = salta fra righe)
  T3  ciascun pezzo piu' fitto: QPxQP coerente gold (nk 448, nquad 160); residuo con
      p x4 e pannelli angolari x4
Uso (dalla radice di SLIM):
  python3 test/spike_probe.py --up UP.npz --down DN.npz --table out/cluster/P0p80_prod_union/snap/iter001.npz
      [--tests 1,2,3] [--wlo -0.05 --whi 0.05 --dw 2e-4] [--workers 10]
Uscita in out/impi_probe/spike_<nome>/: npz per test e un grafico.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import impi_union_probe as R  # noqa: E402  (residuo a griglia, stessa integrazione di impi_table)


def qpqp(up, down, q_list, omegas, nk, nquad, workers, gamma_floor=1.0e-3):
    """Parte QPxQP semi-analitica come in fflo/pairbuild.py (angle_mode kjac)."""
    from concurrent.futures import ProcessPoolExecutor
    from fflo.qp_cube import qp_model_from_cube
    from fflo.impi_cv import _canonical_qp_fields, _cv_init, _cv_work, _kf
    mu_ = qp_model_from_cube(up), qp_model_from_cube(down)
    ku, eu, zu, gu, mu_up = _canonical_qp_fields(up, mu_[0])
    kd, ed, zd, gd, mu_dn = _canonical_qp_fields(down, mu_[1])
    feats = [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)]
    opts = dict(angle_mode="kjac", nk=nk, nphi=64, n_kquad=nquad, kquad_chunk_size=8, kmax=4.0,
                gamma_floor=gamma_floor)
    with ProcessPoolExecutor(max_workers=min(workers, len(q_list)), initializer=_cv_init,
                             initargs=(np.asarray(omegas, float), ku, eu, zu, gu, kd, ed, zd, gd, feats, opts)) as pool:
        return np.asarray(list(pool.map(_cv_work, list(q_list), chunksize=1)))


def build_a0_zfloor(src, dst):
    """A0 come build_qp_cube ma con la larghezza minima Z*eta invece di eta: il polo di Dyson
    con ImSigma = 0 ha semilarghezza Z*(|ImSigma|+eta) >= Z*eta, non eta."""
    from fflo.qp_cube import qp_model_from_cube
    c = dict(np.load(src))
    model = qp_model_from_cube(src)
    k = np.asarray(c["k"], float)
    w2 = c["w"].astype(float) if c["w"].ndim == 2 else None
    wb = np.asarray(c["w_base"] if "w_base" in c else c["w"], float)
    km = np.asarray(model["k"], float)
    E = np.interp(k, km, np.asarray(model["E"], float))
    Z = np.interp(k, km, np.asarray(model["Z"], float))
    G = np.maximum(np.interp(k, km, np.asarray(model["G"], float)), Z * float(model["eta"]))
    out = np.zeros_like(np.asarray(c["A"], float))
    for ik in range(k.size):
        w = w2[ik] if w2 is not None else wb
        out[ik] = (Z[ik] / np.pi) * G[ik] / ((w - E[ik]) ** 2 + G[ik] ** 2)
    c["A"] = out
    np.savez(dst, **c)


def residual(up, down, a0u, a0d, q_list, omegas, workers, **kw):
    R.setup(up, down, a0u, a0d, **kw)
    return R.run(list(q_list), np.asarray(omegas, float), "union", workers)


def spike_metric(w, rows, w0=-0.0095, half=0.004):
    """Ampiezza del picco: valore in [w0-half, w0+half] meno la media dei bordi della finestra."""
    m = np.abs(w - w0) <= half
    edge = (np.abs(w - w0) > half) & (np.abs(w - w0) <= 3 * half)
    return np.array([r[m][np.argmax(np.abs(r[m] - r[edge].mean()))] - r[edge].mean() for r in rows])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--up", required=True)
    ap.add_argument("--down", required=True)
    ap.add_argument("--table", required=True, help="snap iterNNN.npz con le righe spurie (per le Q e qff)")
    ap.add_argument("--tests", default="1,2,3")
    ap.add_argument("--wlo", type=float, default=-0.05)
    ap.add_argument("--whi", type=float, default=0.05)
    ap.add_argument("--dw", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--t3", default="qpqp,dw", help="varianti extra di T3: qpqp (coerente gold), dw (eps fine)")
    a = ap.parse_args(argv)
    tests = {int(t) for t in a.tests.split(",")}
    name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.table))))
    out = a.out or os.path.join(HERE, "out", "impi_probe", f"spike_{name}")
    os.makedirs(out, exist_ok=True)

    from fflo.qp_cube import build_qp_cube, qp_model_from_cube
    a0u, a0d = os.path.join(out, "A0_up.npz"), os.path.join(out, "A0_down.npz")
    for src, dst in ((a.up, a0u), (a.down, a0d)):
        if not os.path.exists(dst):
            build_qp_cube(src, dst, model=qp_model_from_cube(src))

    z = np.load(a.table)
    qtab, qff = np.asarray(z["q"], float), float(z["qff"])
    sel = np.flatnonzero((qtab / qff > 0.96) & (qtab / qff < 1.035))
    q_rows = qtab[sel]
    w = np.arange(a.wlo, a.whi + 0.5 * a.dw, a.dw)
    print(f"[spike] {name}: qff = {qff:.5f}; righe della tabella Q/qff = {np.round(q_rows / qff, 4)}; "
          f"{w.size} Omega in [{a.wlo}, {a.whi}]", flush=True)

    def report(tag, rows, qs):
        sp = spike_metric(w, rows)
        print(f"  {tag:34s} picco a Omega~-0.0095 (x1e-3): " +
              "  ".join(f"{x / qff:.4f}:{v * 1e3:+6.2f}" for x, v in zip(qs, sp)), flush=True)

    res = {}
    if 1 in tests:
        t0 = time.time()
        res["T1_qpqp"] = qpqp(a.up, a.down, q_rows, w, 96, 48, a.workers)
        res["T1_resid"] = residual(a.up, a.down, a0u, a0d, q_rows, w, a.workers)
        print(f"[T1] {time.time() - t0:.0f}s: righe della tabella, turbo", flush=True)
        report("QPxQP (turbo 96/48)", res["T1_qpqp"], q_rows)
        report("residuo a griglia (p31, ang 10+7)", res["T1_resid"], q_rows)
        report("somma", res["T1_qpqp"] + res["T1_resid"], q_rows)
    if 2 in tests:
        t0 = time.time()
        qd = qff * np.arange(0.97, 1.0301, 0.002)
        wn = w[(w > -0.03) & (w < 0.01)]
        res["T2_q"] = qd
        res["T2_qpqp"] = qpqp(a.up, a.down, qd, w, 96, 48, a.workers)
        res["T2_resid"] = residual(a.up, a.down, a0u, a0d, qd, w, a.workers)
        print(f"[T2] {time.time() - t0:.0f}s: Q fitto (passo 0.002 qff)", flush=True)
        report("QPxQP, Q fitto", res["T2_qpqp"], qd)
        report("residuo, Q fitto", res["T2_resid"], qd)
    if 4 in tests:
        # T4: control variate con A0 larga quanto il polo vero (floor Z*eta, anche nella parte analitica)
        t0 = time.time()
        a0uz, a0dz = os.path.join(out, "A0z_up.npz"), os.path.join(out, "A0z_down.npz")
        for src, dst in ((a.up, a0uz), (a.down, a0dz)):
            if not os.path.exists(dst):
                build_a0_zfloor(src, dst)
        res["T4_qpqp_z"] = qpqp(a.up, a.down, q_rows, w, 96, 48, a.workers, gamma_floor=1.0e-5)
        res["T4_resid_z"] = residual(a.up, a.down, a0uz, a0dz, q_rows, w, a.workers)
        print(f"[T4] {time.time() - t0:.0f}s: A0 con floor Z*eta", flush=True)
        report("QPxQP, floor 1e-5", res["T4_qpqp_z"], q_rows)
        report("residuo con A0 floor Z*eta", res["T4_resid_z"], q_rows)
        report("somma (T4)", res["T4_qpqp_z"] + res["T4_resid_z"], q_rows)
    if 3 in tests:
        t0 = time.time()
        if "qpqp" in a.t3:
            res["T3_qpqp_gold"] = qpqp(a.up, a.down, q_rows, w, 448, 160, a.workers)
        if "dw" in a.t3:
            res["T3_resid_dw4"] = residual(a.up, a.down, a0u, a0d, q_rows, w, a.workers, dw=2.5e-4, n_linear=160)
            report("residuo eps dw/4 (2.5e-4, stesso blocco)", res["T3_resid_dw4"], q_rows)
        res["T3_resid_p4"] = residual(a.up, a.down, a0u, a0d, q_rows, w, a.workers, n_p=121)
        res["T3_resid_ang4"] = residual(a.up, a.down, a0u, a0d, q_rows, w, a.workers, n_kquad=40, kquad_feat=28)
        print(f"[T3] {time.time() - t0:.0f}s: piu' fitto", flush=True)
        if "T3_qpqp_gold" in res:
            report("QPxQP gold (448/160)", res["T3_qpqp_gold"], q_rows)
        report("residuo p x4 (121)", res["T3_resid_p4"], q_rows)
        report("residuo angolo x4 (40+28)", res["T3_resid_ang4"], q_rows)
    np.savez(os.path.join(out, "spike_probe.npz"), w=w, q_rows=q_rows, qff=qff, **res)
    plot(os.path.join(out, "spike_probe.npz"))
    return 0


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(path)
    w, q, qff = z["w"], z["q_rows"], float(z["qff"])
    keys = [k for k in ("T1_qpqp", "T1_resid", "T4_qpqp_z", "T4_resid_z", "T3_qpqp_gold", "T3_resid_dw4", "T3_resid_p4", "T3_resid_ang4") if k in z.files]
    if not keys:
        return
    fig, axs = plt.subplots(len(keys), 1, figsize=(10, 2.6 * len(keys)), layout="constrained", squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, k in zip(axs[:, 0], keys):
        for i in range(q.size):
            ax.plot(w, z[k][i] * 1e3, lw=0.9, color=cmap(i / max(1, q.size - 1)), label=f"{q[i] / qff:.4f} qff")
        ax.set_title(k, fontsize=9)
        ax.axvline(-0.0095, color="#6b6b68", lw=0.6, ls=":")
    axs[0, 0].legend(fontsize=7, ncol=4, frameon=False)
    axs[-1, 0].set_xlabel("Ω")
    fig.savefig(path.replace(".npz", ".png"), dpi=110)
    print("scritto", path.replace(".npz", ".png"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
