#!/usr/bin/env python3
"""Test locali sulla (mancata) criticita' di Gamma a Q = qff, dai dati pullati.

  census   ImGamma^-1 == 0 esatto (limitatore di segno) e valori col segno sbagliato vicino a
           Omega = 0 nelle righe 0.9-1.15 qff, dalle snap di tutte le run
  decomp   ImPi(qff, Omega) vicino a 0 dalle cube di una run, pezzo per pezzo:
             QPxQP semi-analitico (impi_cv, kjac) + residuo a griglia (A*A - A0*A0, come impi_table)
           in tre varianti (eta, passo eps) = (1e-3, 1e-3) produzione, (1e-3, 2.5e-4), (2.5e-4, 2.5e-4)
           piu' G0G0 libero (E = k^2/m - mu, Z = 1, Gamma = eta) con eta 1e-3 e 2.5e-4.
           Stampa anche la tangenza delle superfici di Fermi vestite kF_up^QP - kF_dn^QP contro qff.
  delta    risposta a delta a Pi fissato, dalla tabella della snap: si ripinna ReGamma^-1 a -delta'
           e si guardano |Gamma(Q*, Omega = 1e-3)| e il peso di coppia a Omega < 0 sull'anello
Uso (dalla radice di SLIM):
  python3 test/criticality_probe.py census
  python3 test/criticality_probe.py decomp P0p50_prod [--workers 12]
  python3 test/criticality_probe.py delta P0p50_prod P0p30_prod ...
Uscita in out/impi_probe/criticality/.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
BASE = os.path.join(HERE, "out", "cluster")
OUT = os.path.join(HERE, "out", "impi_probe", "criticality")
DELTA = 1e-3
CENSUS_RUNS = ["P0p30_prod", "P0p40_prod", "P0p50_prod", "P0p60_prod", "P0p65_prod",
               "P0p70_prod_union_fine", "P0p75_prod_union_fine", "P0p75_gmax", "P0p80_prod_union",
               "P0p85_prod_union", "P0p90_prod_union"]


def last_snap(run):
    s = sorted(glob.glob(os.path.join(BASE, run, "snap", "iter*.npz")))
    return s[-1] if s else None


# ------------------------------------------------------------------ census
def census(runs):
    for run in runs:
        s = last_snap(run)
        if s is None:
            continue
        z = np.load(s)
        q, w, qff, im = z["q"], z["omega"], float(z["qff"]), z["ImInvGamma"]
        rows = np.flatnonzero((q / qff > 0.9) & (q / qff < 1.15))
        cols = np.flatnonzero((np.abs(w) <= 0.05) & (w != 0))
        blk = im[np.ix_(rows, cols)]
        zero = blk == 0
        wrong = (blk * np.sign(w[cols])[None, :]) < 0
        iq = int(np.argmin(np.abs(q - qff)))
        wz = w[cols][im[iq, cols] == 0]
        rng = f"[{wz.min():+.4f}, {wz.max():+.4f}]" if wz.size else "nessuno"
        print(f"{run:24s} it{s[-7:-4]}  |Omega|<=0.05, righe 0.9-1.15 qff: zeri esatti {100 * zero.mean():5.1f}%"
              f"  segno sbagliato {int(wrong.sum())}  | riga qff: zeri in {rng} ({wz.size} nodi)")


# ------------------------------------------------------------------ decomp
def modified_cube(src, dst, eta, floor_wc=0.0):
    """Copia della cube con eta diverso e A ricostruita dalla stessa Sigma (g = |ImS| + eta).

    floor_wc > 0: toglie il "pavimento" di ImSigma a omega = 0 (a T = 0 ImSigma(k, 0) = 0; la Sigma
    grezza ha nodi in omega a ~7e-3 e il vertice a V non e' risolto):
        ImS -> min(ImS - ImS(k, 0) * max(0, 1 - |w| / floor_wc), 0).   ReS resta com'e'.
    """
    c = dict(np.load(src, allow_pickle=True))
    k = np.asarray(c["k"], float)
    w = np.asarray(c["w"], float)
    re, im = np.asarray(c["ReS"], float), np.asarray(c["ImS"], float)
    w = w if w.ndim == 2 else np.broadcast_to(w, re.shape)
    if floor_wc > 0.0:
        im0 = np.array([np.interp(0.0, w[i], im[i]) for i in range(k.size)])[:, None]
        im = np.minimum(im - im0 * np.maximum(0.0, 1.0 - np.abs(w) / floor_wc), 0.0)
        c["ImS"] = im
    spin_dn = "spindown" in os.path.basename(src)
    mu = float(c["mu_dn"] if spin_dn else c["mu_up"])
    mass = float(c["mass"]) if "mass" in c else 1.0
    g = np.abs(im) + eta
    xi = (k ** 2 / mass - mu)[:, None]
    c["A"] = (1.0 / np.pi) * g / ((w - xi - (re - float(c["sigma0"]))) ** 2 + g ** 2)
    c["eta"] = np.asarray(eta)
    np.savez(dst, **c)


def qpqp_fields(fields_up, fields_dn, q_list, omegas, workers, gamma_floor, nk=96, nquad=48):
    from concurrent.futures import ProcessPoolExecutor
    from fflo.impi_cv import _cv_init, _cv_work, _kf
    ku, eu, zu, gu, mu_up = fields_up
    kd, ed, zd, gd, mu_dn = fields_dn
    feats = [_kf(ku, eu, mu_up), _kf(kd, ed, mu_dn)]
    opts = dict(angle_mode="kjac", nk=nk, nphi=64, n_kquad=nquad, kquad_chunk_size=8, kmax=4.0,
                gamma_floor=gamma_floor)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(q_list))), initializer=_cv_init,
                             initargs=(np.asarray(omegas, float), ku, eu, zu, gu, kd, ed, zd, gd, feats, opts)) as pool:
        return np.asarray(list(pool.map(_cv_work, list(q_list), chunksize=1)))


VARIANTS = {  # tag: (fattore su eta, passo eps, n_linear, floor_wc)
    "prod": (1.0, 1e-3, 40, 0.0), "dwfine": (1.0, 2.5e-4, 160, 0.0), "eta4": (0.25, 2.5e-4, 160, 0.0),
    "fix": (1.0, 2.5e-4, 160, 0.008), "fix4": (0.25, 2.5e-4, 160, 0.008),
}


def decomp(run, workers, q_rel, variants=("prod", "dwfine", "eta4")):
    import impi_union_probe as R
    from fflo.impi_cv import _canonical_qp_fields, _kf
    from fflo.qp_cube import build_qp_cube, qp_model_from_cube
    up = sorted(glob.glob(os.path.join(BASE, run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    out = os.path.join(OUT, run)
    os.makedirs(out, exist_ok=True)
    c = np.load(up, allow_pickle=True)
    mu_up, mu_dn, eta0 = float(c["mu_up"]), float(c["mu_dn"]), float(c["eta"])
    mass = float(c["mass"]) if "mass" in c.files else 1.0
    qff = np.sqrt(mu_up) - np.sqrt(mu_dn)
    q_list = [r * qff for r in q_rel]
    wpos = np.geomspace(1e-4, 0.1, 31)
    omegas = np.concatenate([-wpos[::-1], [0.0], wpos])
    path = os.path.join(out, "decomp.npz")
    res = dict(np.load(path)) if os.path.exists(path) else {}
    res.update(w=omegas, q=np.asarray(q_list), qff=qff)
    # tangenza delle superfici di Fermi vestite
    mdl_u, mdl_d = qp_model_from_cube(up), qp_model_from_cube(dn)
    fu, fd = _canonical_qp_fields(up, mdl_u), _canonical_qp_fields(dn, mdl_d)
    kfu, kfd = _kf(fu[0], fu[1], mu_up), _kf(fd[0], fd[1], mu_dn)
    zf = float(np.interp(kfd, fd[0], fd[2]))
    print(f"[decomp] {run} ({os.path.basename(up)}): qff = {qff:.5f};  kF^QP up {kfu:.5f} (nudo {np.sqrt(mu_up):.5f}), "
          f"dn {kfd:.5f} (nudo {np.sqrt(mu_dn):.5f});  tangenza vestita = {(kfu - kfd) / qff:.5f} qff;  Z_dn(kF) = {zf:.3f}",
          flush=True)
    res.update(kfu=kfu, kfd=kfd, zdn=zf)
    for tag in [v for v in variants if v not in ("free", "free4")]:
        fac, dw, nlin, wc = VARIANTS[tag]
        eta = eta0 * fac
        t0 = time.time()
        if fac == 1.0 and wc == 0.0:
            cu, cd = up, dn
        else:
            cu, cd = os.path.join(out, f"A_komega_spinup_{tag}.npz"), os.path.join(out, f"A_komega_spindown_{tag}.npz")
            modified_cube(up, cu, eta, wc)
            modified_cube(dn, cd, eta, wc)
        a0u, a0d = os.path.join(out, f"A0_{tag}_spinup.npz"), os.path.join(out, f"A0_{tag}_spindown.npz")
        mu_m, md_m = qp_model_from_cube(cu), qp_model_from_cube(cd)
        build_qp_cube(cu, a0u, model=mu_m)
        build_qp_cube(cd, a0d, model=md_m)
        cf = np.load(cd)
        imk = np.interp(0.0, np.asarray(cf["w"])[int(np.argmin(np.abs(cf["k"] - kfd)))],
                        np.asarray(cf["ImS"])[int(np.argmin(np.abs(cf["k"] - kfd)))])
        print(f"[decomp] {tag}: ImS_dn(kF, 0) = {imk:+.2e};  modello QP a kF_dn: Z = "
              f"{np.interp(kfd, md_m['k'], md_m['Z']):.3f}, Gamma_QP = {np.interp(kfd, md_m['k'], md_m['G']):.2e}", flush=True)
        res[f"{tag}_qpqp"] = qpqp_fields(_canonical_qp_fields(cu, mu_m), _canonical_qp_fields(cd, md_m),
                                         q_list, omegas, workers, gamma_floor=eta)
        R.setup(cu, cd, a0u, a0d, dw=dw, n_linear=nlin)
        res[f"{tag}_resid"] = R.run(q_list, omegas, "union", workers)
        print(f"[decomp] {tag}: eta {eta:g}, passo eps {dw:g}: {time.time() - t0:.0f}s", flush=True)
        np.savez(os.path.join(out, "decomp.npz"), **res)
    # G0G0 libero (poli di larghezza eta: quadratura gold)
    k = np.linspace(1e-4, 6.0, 4001)
    one = np.ones_like(k)
    for tag, eta in (("free", eta0), ("free4", eta0 / 4)):
        if tag not in variants:
            continue
        t0 = time.time()
        res[f"{tag}_qpqp"] = qpqp_fields((k, k ** 2 / mass - mu_up, one, eta * one, mu_up),
                                         (k, k ** 2 / mass - mu_dn, one, eta * one, mu_dn),
                                         q_list, omegas, workers, gamma_floor=eta, nk=448, nquad=160)
        print(f"[decomp] {tag}: eta {eta:g}: {time.time() - t0:.0f}s", flush=True)
    s = last_snap(run)
    if s:
        z = np.load(s)
        iq = int(np.argmin(np.abs(z["q"] - qff)))
        res["snap_im"] = np.interp(omegas, z["omega"], z["ImInvGamma"][iq])
    np.savez(path, **res)
    report_decomp(path)
    plot_decomp(path)


def report_decomp(path):
    z = np.load(path)
    w, q, qff = z["w"], z["q"], float(z["qff"])
    xs = (-0.03, -0.01, -0.003, -0.001, -0.0003, 0.0003, 0.001, 0.003, 0.01, 0.03)
    print("  ImGamma^-1 = -ImPi, x1e3 (unita' di delta)")
    for i in range(q.size):
        print(f"  Q = {q[i] / qff:.3f} qff;  Omega:  " + " ".join(f"{x:+8.4f}" for x in xs))
        for tag in ("prod", "dwfine", "eta4", "fix", "fix4"):
            if f"{tag}_qpqp" not in z.files:
                continue
            for part in ("qpqp", "resid"):
                r = -z[f"{tag}_{part}"][i]
                print(f"    {tag:7s} {part:6s}" + " ".join(f"{np.interp(x, w, r) * 1e3:+8.3f}" for x in xs))
            r = -(z[f"{tag}_qpqp"][i] + z[f"{tag}_resid"][i])
            print(f"    {tag:7s} {'somma':6s}" + " ".join(f"{np.interp(x, w, r) * 1e3:+8.3f}" for x in xs))
        for tag in ("free", "free4"):
            if f"{tag}_qpqp" in z.files:
                r = -z[f"{tag}_qpqp"][i]
                print(f"    {tag:7s} {'G0G0':6s}" + " ".join(f"{np.interp(x, w, r) * 1e3:+8.3f}" for x in xs))
        if "snap_im" in z.files and i == 0:
            print(f"    {'snap':7s} {'tab.':6s}" + " ".join(f"{np.interp(x, w, z['snap_im']) * 1e3:+8.3f}" for x in xs))


def plot_decomp(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(path)
    w, q, qff = z["w"], z["q"], float(z["qff"])
    fig, axs = plt.subplots(q.size, 2, figsize=(12, 3.8 * q.size), layout="constrained", squeeze=False)
    sty = {"prod": ("#eb6834", "-"), "dwfine": ("#2a78d6", "-"), "eta4": ("#1baf7a", "-"),
           "fix": ("#4a3aa7", "-"), "fix4": ("#e87ba4", "-"),
           "free": ("#6b6b68", "--"), "free4": ("#1f1f1e", ":")}
    for i in range(q.size):
        for c, sgn in enumerate((1, -1)):
            ax = axs[i, c]
            m = sgn * w > 0
            for tag, (col, ls) in sty.items():
                if f"{tag}_qpqp" not in z.files:
                    continue
                r = -(z[f"{tag}_qpqp"][i] + (z[f"{tag}_resid"][i] if f"{tag}_resid" in z.files else 0.0))
                y = sgn * r[m]
                ax.loglog(np.abs(w[m]), np.where(y > 0, y, np.nan), ls, color=col, lw=1.1, label=tag)
                ax.loglog(np.abs(w[m]), np.where(y < 0, -y, np.nan), "x", color=col, ms=3)
            if "snap_im" in z.files and i == 0:
                y = sgn * z["snap_im"][m]
                ax.loglog(np.abs(w[m]), np.where(y > 0, y, np.nan), "o", color="#e87ba4", ms=2.5, label="snap")
            xs = np.array([1e-4, 0.1])
            ax.loglog(xs, 3e-3 * (xs / 0.01) ** 0.5, ":", color="#6b6b68", lw=0.7, label="∝ |Ω|^½")
            ax.set(title=f"Q = {q[i] / qff:.3f} qff, Ω {'>' if sgn > 0 else '<'} 0  (x = segno sbagliato)",
                   xlabel="|Ω|", ylabel="|ImΓ⁻¹| = |ImΠ|")
            ax.legend(fontsize=7, frameon=False)
    fig.savefig(path.replace(".npz", ".png"), dpi=115)
    print("scritto", path.replace(".npz", ".png"))


# ------------------------------------------------------------------ sharp
def sharp_qpqp(fu, fd, Q, omegas, nphi=3000, nk=6000, kmax=3.0):
    """QPxQP con poli netti (eta -> 0): ImGamma^-1 = (I_pp - I_hh) / (4 pi),
    I = int d^2k Z_up(k) Z_dn(|Q-k|) delta(Omega - E_up(k) - E_dn(|Q-k|)) sulle radici lungo i raggi.
    Normalizzazione: nel vuoto I = pi/2 -> ImPi = -1/8, come ImPiVacuum delle tabelle."""
    (ku, eu, zu), (kd, ed, zd) = fu, fd
    phi = (np.arange(nphi) + 0.5) * np.pi / nphi          # [0, pi], simmetria phi -> -phi
    k = np.linspace(1e-5, kmax, nk)
    dk = k[1] - k[0]
    kq = np.sqrt(np.maximum(Q * Q + k[None, :] ** 2 - 2 * Q * k[None, :] * np.cos(phi)[:, None], 0.0))
    Eu = np.interp(k, ku, eu)[None, :] * np.ones((nphi, 1))
    Ed = np.interp(kq, kd, ed)
    Zp = np.interp(k, ku, zu)[None, :] * np.interp(kq, kd, zd)
    S = Eu + Ed
    out = np.zeros(len(omegas))
    for j, om in enumerate(omegas):
        F = S - om
        ia, ib = np.nonzero(np.signbit(F[:, :-1]) != np.signbit(F[:, 1:]))
        f0, f1 = F[ia, ib], F[ia, ib + 1]
        t = f0 / (f0 - f1)
        kr = k[ib] + t * dk
        slope = np.abs(f1 - f0) / dk
        eu_r = Eu[ia, ib] + t * (Eu[ia, ib + 1] - Eu[ia, ib])
        ed_r = Ed[ia, ib] + t * (Ed[ia, ib + 1] - Ed[ia, ib])
        z_r = Zp[ia, ib] + t * (Zp[ia, ib + 1] - Zp[ia, ib])
        sgn = np.where((eu_r > 0) & (ed_r > 0), 1.0, np.where((eu_r < 0) & (ed_r < 0), -1.0, 0.0))
        I = np.sum(sgn * z_r * kr / slope) * (np.pi / nphi) * 2.0
        out[j] = I / (4 * np.pi)
    return out


def sharp(run):
    from fflo.qp_cube import qp_model_from_cube
    up = sorted(glob.glob(os.path.join(BASE, run, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    dn = up.replace("spinup", "spindown")
    c = np.load(up, allow_pickle=True)
    mu_up, mu_dn = float(c["mu_up"]), float(c["mu_dn"])
    mass = float(c["mass"]) if "mass" in c.files else 1.0
    qff = np.sqrt(mu_up) - np.sqrt(mu_dn)
    wpos = np.geomspace(1e-4, 0.1, 31)
    omegas = np.concatenate([-wpos[::-1], wpos])
    kk = np.linspace(1e-5, 4.0, 40001)
    one = np.ones_like(kk)
    t0 = time.time()
    free = sharp_qpqp((kk, kk ** 2 / mass - mu_up, one), (kk, kk ** 2 / mass - mu_dn, one), qff, omegas)
    mu_m, md_m = qp_model_from_cube(up), qp_model_from_cube(dn)
    dress = sharp_qpqp((mu_m["k"], mu_m["E"], mu_m["Z"]), (md_m["k"], md_m["E"], md_m["Z"]), qff, omegas)
    print(f"[sharp] {run}: {time.time() - t0:.0f}s   ImGamma^-1(qff) x1e3 con poli netti")
    xs = (-0.03, -0.01, -0.003, -0.001, -0.0003, 0.0003, 0.001, 0.003, 0.01, 0.03)
    print("    Omega:    " + " ".join(f"{x:+8.4f}" for x in xs))
    print("    liberi    " + " ".join(f"{np.interp(x, omegas, free) * 1e3:+8.3f}" for x in xs))
    print("    vestiti   " + " ".join(f"{np.interp(x, omegas, dress) * 1e3:+8.3f}" for x in xs))
    out = os.path.join(OUT, run)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "sharp.npz"), w=omegas, free=free, dress=dress, qff=qff)
    return omegas, free, dress


# ------------------------------------------------------------------ model
def model_exponent(mu_u=1.5, mu_d=0.5):
    """Bolla a Q = qff (tangenza): maggioritario netto, minoritario libero con larghezza Gamma(nu) = g|nu|^alpha.
    Verifica ImPi(qff, Omega) ~ Omega^(1 - alpha/2): alpha = 1 -> 1/2, alpha = 2/3 -> 2/3, alpha = 1/2 -> 3/4."""
    ku, kd = np.sqrt(mu_u), np.sqrt(mu_d)
    Q = ku - kd
    th = np.concatenate([-np.geomspace(1e-7, np.pi, 6000)[::-1], [0.0], np.geomspace(1e-7, np.pi, 6000)])
    xg, wg = np.polynomial.legendre.leggauss(160)

    def impi(om, alpha, g):
        u = 0.5 * (xg + 1)
        xi = om * u * u * (3 - 2 * u)
        wxi = om * wg * 3 * u * (1 - u)
        k = np.sqrt(mu_u + xi)[:, None]
        p = np.sqrt(Q * Q + k * k - 2 * Q * k * np.cos(th)[None, :])
        nu = (om - xi)[:, None]
        G = g * np.abs(nu) ** alpha
        A = (1 / np.pi) * G / ((nu - (p * p - mu_d)) ** 2 + G * G)
        return np.sum(wxi * 0.5 * np.trapezoid(A, th, axis=1))

    oms = np.geomspace(1e-5, 1e-2, 13)
    for alpha, g in ((1.0, 0.5), (2 / 3, 0.1), (0.5, 0.05)):
        v = np.array([impi(o, alpha, g) for o in oms])
        sl = np.diff(np.log(v)) / np.diff(np.log(oms))
        print(f"alpha = {alpha:.3f}: esponente locale 1e-5 -> 1e-2: " + " ".join(f"{s:.3f}" for s in sl)
              + f"   atteso {1 - alpha / 2:.3f}")


# ------------------------------------------------------------------ delta
def delta_response(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dls = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0.0])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for run in runs:
        s = last_snap(run)
        if s is None:
            continue
        z = np.load(s)
        q, w, qff = z["q"], z["omega"], float(z["qff"])
        re, im = z["ReInvGamma"], z["ImInvGamma"]
        i0 = int(np.argmin(np.abs(w)))
        m = q <= 2 * qff
        iq = int(np.argmax(np.where(m, re[:, i0], -np.inf)))
        base = re - re[iq, i0]                     # pinnata a 0: ReGamma^-1 = base - delta'
        neg = (w < 0) & (w > -1.0)
        ring = ((q / qff > 0.8) & (q / qff < 1.25)).astype(float)
        gam, wt = [], []
        for d in dls:
            G = 1.0 / (base - d + 1j * im)
            gam.append(abs(float(np.interp(1e-3, w, G[iq].imag))))
            wt.append(np.trapezoid(np.trapezoid(np.abs(G.imag[:, neg]), w[neg], axis=1) * q * ring, q))
        gam, wt = np.array(gam), np.array(wt)
        P = run.split("_")[0].replace("P0p", "0.")
        d0 = (np.interp(0.005, w, im[iq]) - np.interp(-0.005, w, im[iq])) / DELTA
        print(f"[delta] {run} it{s[-7:-4]}  Q* = {q[iq] / qff:.3f} qff, gradino D = {d0:.1f} delta")
        print("    delta'             " + " ".join(f"{d:8.0e}" for d in dls))
        print("    |ImGamma(Q*,1e-3)| " + " ".join(f"{x:8.1f}" for x in gam))
        print("    peso anello w<0    " + " ".join(f"{x / wt[2]:8.3f}" for x in wt) + "   (rispetto a delta' = 1e-3)")
        xd = np.where(dls > 0, dls, 3e-5)
        axs[0].loglog(xd, gam, "-o", ms=3, label=f"P = {P} (D = {d0:.1f} δ)")
        axs[1].semilogx(xd, wt / wt[2], "-o", ms=3, label=f"P = {P}")
    axs[0].loglog([1e-4, 1e-2], [1e4, 1e2], ":", color="#6b6b68", label="∝ 1/δ' (critico)")
    axs[0].set(xlabel="δ'  (0 disegnato a 3e-5)", ylabel="|ImΓ(Q*, Ω = 1e-3)|", title="Γ al pin, a Π fissato")
    axs[1].set(xlabel="δ'", ylabel="peso a Ω<0 sull'anello / valore a δ' = 1e-3", title="peso di coppia, a Π fissato")
    for ax in axs:
        ax.legend(fontsize=7, frameon=False)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "delta_response.png"), dpi=115)
    print("scritto", os.path.join(OUT, "delta_response.png"))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("census", "decomp", "delta", "report", "sharp", "model"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--q", default="1.0", help="righe di Q in unita' di qff (decomp)")
    ap.add_argument("--variants", default="prod,dwfine,eta4,free,free4",
                    help="decomp: fra " + ",".join(list(VARIANTS) + ["free", "free4"]))
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.what == "census":
        census(a.runs or CENSUS_RUNS)
    elif a.what == "delta":
        delta_response(a.runs or ["P0p30_prod", "P0p50_prod", "P0p65_prod", "P0p70_prod_union_fine",
                                  "P0p75_prod_union_fine", "P0p90_prod_union"])
    elif a.what == "model":
        model_exponent()
    elif a.what == "sharp":
        for run in a.runs:
            sharp(run)
    elif a.what == "report":
        for run in a.runs:
            report_decomp(os.path.join(OUT, run, "decomp.npz"))
            plot_decomp(os.path.join(OUT, run, "decomp.npz"))
    else:
        for run in a.runs:
            decomp(run, a.workers, [float(x) for x in a.q.split(",")], a.variants.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
