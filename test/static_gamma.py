#!/usr/bin/env python3
"""ReGamma^-1(Q, 0) STATICO dalle cube: niente griglia in Omega, niente KK, niente taper, niente A0 (2026-10-05).

A T = 0, per le funzioni di Green della cube (Sigma prolungata sull'asse immaginario):

    ReGamma^-1(Q, 0) = 1/g + S(Q),     S(Q) = int d^2p/(2pi)^2 K(p, |Q - p|) = -RePi(Q, 0)
    K(k1, k2) = int dw/(2pi) G_up(k1, iw) G_dn(k2, -iw) = (1/pi) int_0^inf Re[G_up(k1, iw) conj(G_dn(k2, iw))] dw
              ( = int int A_up A_dn [th(e) th(e') - th(-e) th(-e')] / (e + e') )
    G(k, iw) = 1 / (iw - (k^2 - mu) - (Sigma(k, iw) - sigma0)),   (sigma0 come nella cube, density.py:188)   Sigma(k, iw) = Sigma_c(k) + int rho(e) / (iw - e) de

con rho = -ImSigma/pi sulle righe della cube (integrazione esatta di rho lineare a tratti) e Sigma_c dal
confronto ReSigma - KK[ImSigma] sulla riga.  Sull'asse immaginario G e' liscia: K si ottiene con una
somma su una griglia logaritmica in w (prodotto di matrici).  L'integrale in p e' sulla griglia fine
(righe della cube + finestre attorno a kF vestiti e liberi); l'angolo e' integrato in forma chiusa con K
lineare in u = |Q - p|^2 (la tangenza FFLO, dove lo jacobiano diverge, e' esatta).

Il confronto con la produzione usa la differenza con la bolla libera sulla STESSA quadratura:
    D_Lambda(Q) = S_num,Lambda(Q) - S_free,Lambda(Q)  ->  converge per Lambda -> inf (niente taper),
e ReGamma^-1(Q, 0) = Gamma0^-1(Q, 0) + D_inf(Q) + [S_free - S_ref,eta](Q), l'ultimo termine dal confronto
fra la libera esatta e la Gamma0^-1 della tabella (eta del riferimento).

Uso (dalla radice di SLIM):
    python3 test/static_gamma.py free                         # libera: quadratura contro integrale adattivo
    python3 test/static_gamma.py run CUBEDIR [--table T.npz]  # cube vestite (+ confronto con una pair table)
La produzione e' solo letta.
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
from fflo.pair_gamma import kk_pv_linear_at  # noqa: E402

DELTA = 1e-3


# ---------------------------------------------------------------- frequenze di Matsubara (T = 0, continue)
def omega_grid(wmin=1e-9, wmax=1e5, per_decade=40):
    n = int(round(np.log10(wmax / wmin) * per_decade)) + 1
    w = np.geomspace(wmin, wmax, n)
    lw = np.log(w)
    # trapezio in ln w: int f dw = int f w dlnw
    dl = np.diff(lw)
    wt = np.zeros_like(w)
    wt[:-1] += 0.5 * dl
    wt[1:] += 0.5 * dl
    wt *= w
    wt[0] += w[0]              # [0, wmin]: f ~ costante
    return w, wt


def kernel(g1, g2, wt, wmax):
    """K(k1, k2) = (1/pi) int_0^inf Re[G1 conj(G2)] dw, piu' la coda 1/w^2 oltre wmax."""
    m = (g1 * wt[None, :]) @ np.conj(g2).T
    return m.real / np.pi + 1.0 / (np.pi * wmax)


# ---------------------------------------------------------------- Sigma sull'asse immaginario
def sigma_iw_rows(w_rows, ims_rows, iw, chunk=8):
    """int rho(e)/(iw - e) de, rho = -ImS/pi lineare a tratti su ogni riga (forma chiusa per segmento)."""
    nr = w_rows.shape[0]
    out = np.empty((nr, iw.size), dtype=complex)
    z = 1j * iw[None, :]
    for r0 in range(0, nr, chunk):
        for r in range(r0, min(nr, r0 + chunk)):
            x = w_rows[r]
            rho = -ims_rows[r] / np.pi
            a = x[:-1, None]
            b = x[1:, None]
            ra = rho[:-1, None]
            s = ((rho[1:] - rho[:-1]) / (x[1:] - x[:-1]))[:, None]
            L = np.log(z - a) - np.log(z - b)
            seg = ra * L + s * ((z - a) * L - (b - a))
            out[r] = seg.sum(axis=0)
    return out


def sigma_const(w_rows, res_rows, ims_rows, wwin=1.0):
    """ReS(w) - KK[ImS](w) = c0 + c1 w + resto (la KK di produzione e' troncata: il resto e' liscio).
    Fit lineare pesato verso w = 0 su |w| < wwin: c0 + c1 (iw) e' il suo prolungamento, e ReS al livello
    di Fermi (dove sta la superficie di Fermi della cube) e' riprodotto esattamente."""
    nr = w_rows.shape[0]
    c0 = np.empty(nr)
    c1 = np.empty(nr)
    resid = np.empty(nr)
    for r in range(nr):
        x = w_rows[r]
        t = x[np.abs(x) < wwin]
        t = t[(t > x[0]) & (t < x[-1])]
        kk = kk_pv_linear_at(x, ims_rows[r], t)
        cc = np.interp(t, x, res_rows[r]) - kk
        wgt = 1.0 / (np.abs(t) + 1e-3)
        A = np.vstack([np.ones_like(t), t]).T * wgt[:, None]
        sol, *_ = np.linalg.lstsq(A, cc * wgt, rcond=None)
        c0[r], c1[r] = sol
        resid[r] = np.max(np.abs(cc - (sol[0] + sol[1] * t))[np.abs(t) < 0.2]) if np.any(np.abs(t) < 0.2) else np.nan
    return c0, c1, resid


# ---------------------------------------------------------------- integrale in impulso
def angular(Kp, u, p, Q):
    """2 int_0^pi K(p, k2(phi)) dphi con K lineare in u = k2^2 sui nodi u (Kp: valori sulla riga p)."""
    if Q < 1e-12:
        return 2.0 * np.pi * np.interp(p * p, u, Kp)
    c = (p * p + Q * Q - u) / (2.0 * p * Q)
    phi = np.arccos(np.clip(c, -1.0, 1.0))           # decrescente con... u crescente -> phi crescente
    pa, pb = phi[:-1], phi[1:]
    ua, ub = u[:-1], u[1:]
    Ka, Kb = Kp[:-1], Kp[1:]
    dphi = pb - pa
    slope = (Kb - Ka) / (ub - ua)
    int_u = (p * p + Q * Q) * dphi - 2.0 * p * Q * (np.sin(pb) - np.sin(pa))
    seg = Ka * dphi + slope * (int_u - ua * dphi)
    return 2.0 * seg.sum()


def p_panels(breaks, lam_max, nper=20, maxlen=0.2):
    """Nodi e pesi in p su [0, lam_max]: pannelli fra i punti di rottura (kF, |Q -+ kF_dn|, Lambda), Gauss-Legendre
    nella variabile s con p = a + (b - a)(3 s^2 - 2 s^3): lo jacobiano 6 s (1 - s) assorbe le singolarita'
    integrabili 1/sqrt|p - p*| ai bordi dei pannelli (la tangenza FFLO ne mette una a kF_up)."""
    b = np.unique(np.concatenate(([0.0, lam_max], [x for x in breaks if 0.0 < x < lam_max])))
    xs, ws = np.polynomial.legendre.leggauss(nper)
    xs = 0.5 * (xs + 1.0)
    ws = 0.5 * ws
    P, W = [], []
    for a0, b0 in zip(b[:-1], b[1:]):
        m = max(1, int(np.ceil((b0 - a0) / maxlen)))
        edges = np.linspace(a0, b0, m + 1)
        for a1, b1 in zip(edges[:-1], edges[1:]):
            P.append(a1 + (b1 - a1) * (3 * xs ** 2 - 2 * xs ** 3))
            W.append(ws * (b1 - a1) * 6 * xs * (1 - xs))
    return np.concatenate(P), np.concatenate(W)


def S_of_Q(gup_fn, gdn_fine, kg, Q, lams, wt, wmax, breaks, nper=20):
    """S_Lambda(Q) = (1/(2pi)^2) int_0^Lambda p dp 2 int_0^pi dphi K(p, |Q - p|), per ogni Lambda in lams."""
    lam_max = max(lams)
    p, wp = p_panels(list(breaks) + list(lams), lam_max, nper=nper)
    u = kg * kg
    phi = np.empty(p.size)
    for i0 in range(0, p.size, 256):
        pp = p[i0:i0 + 256]
        Krows = kernel(gup_fn(pp), gdn_fine, wt, wmax)
        for j, pv in enumerate(pp):
            phi[i0 + j] = angular(Krows[j], u, pv, Q)
    f = p * phi * wp
    return {L: f[p <= L + 1e-12].sum() / (2.0 * np.pi) ** 2 for L in lams}


def breaks_for(Q, kfu_list, kfd_list):
    out = list(kfu_list)
    for kd in kfd_list:
        out += [Q + kd, abs(Q - kd)]
    return [x for x in out if x > 1e-9]


def fine_grid(k_rows, centers, kmax, lams, dk=0.02, nwin=60):
    parts = [k_rows[k_rows <= kmax], np.linspace(0.0, kmax, int(kmax / dk) + 1), np.asarray(lams, float)]
    for c in centers:
        if not np.isfinite(c) or c <= 0:
            continue
        off = np.concatenate(([0.0], np.geomspace(1e-7, 0.08, nwin)))
        parts += [c - off, c + off]
    g = np.unique(np.concatenate(parts))
    g = g[(g >= 0.0) & (g <= kmax)]
    # niente nodi quasi coincidenti
    keep = np.concatenate(([True], np.diff(g) > 1e-9))
    return g[keep]


# ---------------------------------------------------------------- libera: controllo con integrale adattivo
def free_exact(Q, mu_up, mu_dn, lam):
    from scipy import integrate
    kfu = np.sqrt(mu_up)

    def phi_int(p):
        x1 = p * p - mu_up
        A = p * p + Q * Q - mu_dn
        B = 2.0 * p * Q
        if B < 1e-14:
            x2 = A
            v = (1.0 / (x1 + x2)) if (x1 > 0 and x2 > 0) else ((-1.0 / (x1 + x2)) if (x1 < 0 and x2 < 0) else 0.0)
            return 2.0 * np.pi * v
        cstar = A / B          # x2 = 0 a cos(phi) = cstar

        def f(ph):
            x2 = A - B * np.cos(ph)
            if x1 > 0 and x2 > 0:
                return 1.0 / (x1 + x2)
            if x1 < 0 and x2 < 0:
                return -1.0 / (x1 + x2)
            return 0.0
        pts = []
        if -1.0 < cstar < 1.0:
            pts.append(float(np.arccos(cstar)))
        v, _ = integrate.quad(f, 0.0, np.pi, points=pts or None, limit=400, epsabs=1e-13, epsrel=1e-11)
        return 2.0 * v

    brk = [kfu, abs(Q - np.sqrt(mu_dn)), Q + np.sqrt(mu_dn)]
    brk = sorted(b for b in brk if 0 < b < lam)
    v, _ = integrate.quad(lambda p: p * phi_int(p), 0.0, lam, points=brk or None, limit=800,
                          epsabs=1e-12, epsrel=1e-10)
    return v / (2.0 * np.pi) ** 2


def cmd_free(a):
    mu_up, mu_dn = 1.0 + a.P, 1.0 - a.P
    qff = np.sqrt(mu_up) - np.sqrt(mu_dn)
    w, wt = omega_grid()
    lams = [2.0, 4.0]
    kg = fine_grid(np.array([0.0]), [np.sqrt(mu_up), np.sqrt(mu_dn)], 6.0, lams)
    iw = 1j * w[None, :]
    gup_fn = lambda pp: 1.0 / (iw - (np.asarray(pp)[:, None] ** 2 - mu_up))
    gu = gup_fn(kg)
    gd = 1.0 / (iw - (kg[:, None] ** 2 - mu_dn))
    t0 = time.time()
    K = kernel(gu, gd, wt, w[-1])
    print(f"P={a.P}  griglia fine {kg.size} nodi, {w.size} frequenze, K in {time.time() - t0:.1f} s")
    # controllo puntuale di K contro la forma chiusa
    x1 = kg[:, None] ** 2 - mu_up
    x2 = kg[None, :] ** 2 - mu_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        K0 = np.where((x1 > 0) & (x2 > 0), 1 / (x1 + x2), np.where((x1 < 0) & (x2 < 0), -1 / (x1 + x2), 0.0))
    m = np.isfinite(K0) & (np.abs(x1) > 1e-4) & (np.abs(x2) > 1e-4)
    print(f"   K puntuale: max |K - K_esatto| / K_esatto = {np.max(np.abs(K - K0)[m] / np.maximum(np.abs(K0[m]), 1e-3)):.2e}")
    for Q in (0.0, 0.5 * qff, qff, 1.02 * qff, 2.0):
        s = S_of_Q(gup_fn, gd, kg, Q, lams, wt, w[-1], breaks_for(Q, [np.sqrt(mu_up)], [np.sqrt(mu_dn)]))
        for L in lams:
            ex = free_exact(Q, mu_up, mu_dn, L)
            print(f"   Q/qff={Q / qff:6.3f}  Lambda={L}:  S_quad={s[L]:+.8f}  S_esatto={ex:+.8f}  "
                  f"diff={(s[L] - ex) / DELTA:+.4f} delta")
    return 0


# ---------------------------------------------------------------- vestita
def load_cube(path):
    c = np.load(path)
    return dict(k=np.asarray(c["k"], float), w=np.asarray(c["w"], float), ReS=np.asarray(c["ReS"], float),
                ImS=np.asarray(c["ImS"], float), mu_up=float(c["mu_up"]), mu_dn=float(c["mu_dn"]),
                eta=float(c["eta"]), sigma0=float(c["sigma0"]))


def dressed_kf(cb, mu):
    k = cb["k"]
    re0 = np.array([np.interp(0.0, cb["w"][r], cb["ReS"][r]) for r in range(k.size)])
    e = k * k - mu + re0 - cb["sigma0"]         # la cube usa ReSigma - sigma0 (modo di Fermi, density.py:188)
    idx = np.flatnonzero(np.diff(np.sign(e)) != 0)
    out = []
    for i in idx:
        out.append(k[i] - e[i] * (k[i + 1] - k[i]) / (e[i + 1] - e[i]))
    return out


def cmd_run(a):
    up = glob.glob(os.path.join(a.cubedir, "A_komega_spinup*.npz"))[0]
    dn = glob.glob(os.path.join(a.cubedir, "A_komega_spindown*.npz"))[0]
    cu, cd = load_cube(up), load_cube(dn)
    if a.row_step > 1:          # prova: righe di Sigma decimate (tenendo la prima e l'ultima)
        keep = np.unique(np.concatenate((np.arange(0, cu["k"].size, a.row_step), [cu["k"].size - 1])))
        for cb in (cu, cd):
            for key in ("k", "w", "ReS", "ImS"):
                cb[key] = cb[key][keep]
    mu_up, mu_dn = cu["mu_up"], cu["mu_dn"]
    qff = np.sqrt(mu_up) - np.sqrt(mu_dn)
    w, wt = omega_grid(per_decade=a.per_decade)
    t0 = time.time()
    sig = {}
    for name, cb in (("up", cu), ("dn", cd)):
        c, c1, spread = sigma_const(cb["w"], cb["ReS"], cb["ImS"])
        s_iw = sigma_iw_rows(cb["w"], cb["ImS"], w)
        sig[name] = (c, c1, s_iw)
        kfs = dressed_kf(cb, cb["mu_up"] if name == "up" else cb["mu_dn"])
        big = cb["k"] <= 8.0
        print(f"[{name}] ReS - KK[ImS] = c0 + c1 w: c1 mediana {np.median(c1[big]):+.2e} max|c1| {np.max(np.abs(c1[big])):.2e}; "
              f"resto in |w|<0.2 mediana {np.nanmedian(spread[big]):.1e} max {np.nanmax(spread[big]):.1e} (k<=8);  kF vestiti {np.round(kfs, 6)}  ({time.time() - t0:.0f} s)")
    kfu_d = dressed_kf(cu, mu_up)
    kfd_d = dressed_kf(cd, mu_dn)
    lams = [float(x) for x in a.lams.split(":")]
    kmax = max(lams) + max(2.0, 2.2 * qff) + 0.5
    kg = fine_grid(cu["k"], list(kfu_d) + list(kfd_d) + [np.sqrt(mu_up), np.sqrt(mu_dn)], kmax, lams,
                   dk=a.dk, nwin=a.nwin)
    iw = 1j * w[None, :]

    def g_at(cb, name, mu, eta, kk):
        c, c1, s_iw = sig[name]
        k = cb["k"]
        kk = np.asarray(kk, float)
        sre = np.array([np.interp(kk, k, s_iw[:, j].real) for j in range(w.size)]).T
        sim = np.array([np.interp(kk, k, s_iw[:, j].imag) for j in range(w.size)]).T
        cc = np.interp(kk, k, c) - cb["sigma0"]   # A della cube: w - xi - (ReSigma - sigma0)
        c1k = np.interp(kk, k, c1)
        return 1.0 / (iw + 1j * eta - (kk[:, None] ** 2 - mu) - (cc[:, None] + c1k[:, None] * iw + sre + 1j * sim))

    gdn = {eta: g_at(cd, "dn", mu_dn, eta, kg) for eta in ((0.0,) if a.no_eta else (0.0, a.eta_alt))}
    gup_fn = {eta: (lambda pp, eta=eta: g_at(cu, "up", mu_up, eta, pp)) for eta in (0.0, a.eta_alt)}
    g0d = 1.0 / (iw - (kg[:, None] ** 2 - mu_dn))
    g0u_fn = lambda pp: 1.0 / (iw - (np.asarray(pp)[:, None] ** 2 - mu_up))
    print(f"griglia fine {kg.size} nodi su [0, {kmax:.2f}], {w.size} frequenze ({time.time() - t0:.0f} s)")

    fr = [float(x) for x in a.qfracs.split(":")]
    qs = [f * qff for f in fr] + [float(x) for x in a.extra_q.split(":") if x]
    kfu_all = list(kfu_d) + [np.sqrt(mu_up)]
    kfd_all = list(kfd_d) + [np.sqrt(mu_dn)]
    rows = []
    for Q in qs:
        br = breaks_for(Q, kfu_all, kfd_all)
        s_num = S_of_Q(gup_fn[0.0], gdn[0.0], kg, Q, lams, wt, w[-1], br, nper=a.nper)
        s_eta = (S_of_Q(gup_fn[a.eta_alt], gdn[a.eta_alt], kg, Q, lams, wt, w[-1], br, nper=a.nper)
                 if not a.no_eta else s_num)
        s_free = S_of_Q(g0u_fn, g0d, kg, Q, lams, wt, w[-1], br, nper=a.nper)
        rows.append((Q, s_num, s_eta, s_free))
        print(f"   Q/qff={Q / qff:6.3f} fatto ({time.time() - t0:.0f} s)", flush=True)
    print("\nD_Lambda(Q) = S_num - S_free sulla stessa quadratura [unita' delta = 1e-3]; colonne Lambda = " + str(lams))
    for Q, sn, se, sf in rows:
        print(f"  Q/qff={Q / qff:6.3f}: " + " ".join(f"{(sn[L] - sf[L]) / DELTA:+9.2f}" for L in lams) +
              f"   | eta={a.eta_alt:g} a Lambda max: {(se[lams[-1]] - sf[lams[-1]]) / DELTA:+9.2f}")
    iq = int(np.argmin([abs(r[0] - qff) for r in rows]))
    print("\ngara (D(Q) - D(qff)) per Lambda:")
    for Q, sn, se, sf in rows:
        print(f"  Q/qff={Q / qff:6.3f}: " + " ".join(
            f"{((sn[L] - sf[L]) - (rows[iq][1][L] - rows[iq][3][L])) / DELTA:+8.2f}" for L in lams))
    out = dict(q=np.array([r[0] for r in rows]), lams=np.array(lams), qff=qff,
               S_num=np.array([[r[1][L] for L in lams] for r in rows]),
               S_eta=np.array([[r[2][L] for L in lams] for r in rows]),
               S_free=np.array([[r[3][L] for L in lams] for r in rows]), kg=kg)

    # ReGamma^-1 esatta: Gamma0^-1(Q, 0 + i0) analitica (fflo.proxy) + D_inf (estrapolazione 1/Lambda^2)
    from fflo.proxy import Gamma as gamma_free, configuration as free_conf
    eps0 = float(np.load(up)["eps0"])
    free_conf.update(epsilon_0=eps0, mu_up=mu_up, mu_down=mu_dn)
    g0 = np.array([np.real(1.0 / gamma_free(float(Q), complex(0.0, 1e-10))) for Q in out["q"]])
    L1, L2 = lams[-2], lams[-1]

    def d_inf(Snum):
        Dl = Snum - out["S_free"]
        return Dl[:, -1] + (Dl[:, -1] - Dl[:, -2]) * (1 / L2 ** 2) / (1 / L1 ** 2 - 1 / L2 ** 2)
    ex0 = g0 + d_inf(out["S_num"])
    exe = g0 + d_inf(out["S_eta"])
    out.update(gamma0_inv=g0, re_inv_exact=ex0, re_inv_exact_eta=exe)
    tabs = [(tp, np.load(tp)) for tp in (a.table or [])]
    print("\nReGamma^-1(Q, 0) ESATTA (Gamma0 a eta -> 0 + D_inf) e tabelle [scarto in delta]:")
    print("   Q/qff    Gamma0^-1    esatta(eta=0)  esatta(eta cube)   " + "   ".join(os.path.basename(os.path.dirname(os.path.dirname(tp))) + "/" + os.path.basename(os.path.dirname(tp)) for tp, _ in tabs))
    for i, Q in enumerate(out["q"]):
        cols = []
        for tp, t in tabs:
            tq = np.asarray(t["q"], float)
            j = int(np.argmin(np.abs(tq - Q)))
            if abs(tq[j] - Q) > 1e-6 * max(1.0, Q):
                cols.append("      -       ")
                continue
            v = float(t["ReInvGamma"][j, int(np.argmin(np.abs(np.asarray(t["omega"], float))))])
            cols.append(f"{v:+.5f} ({(v - ex0[i]) / DELTA:+6.1f})")
        print(f"   {Q / qff:6.3f}   {g0[i]:+.5f}    {ex0[i]:+.5f}      {exe[i]:+.5f}      " + "   ".join(cols))
    iz = int(np.argmin(np.abs(out["q"])))
    iq = int(np.argmin(np.abs(out["q"] - qff)))
    gc = lambda v: -4.0 * np.pi * v - 0.5 * np.log(2.0)
    print(f"\nESATTA: g_c(pin qff) = {gc(ex0[iq]):+.4f} (eta cube {gc(exe[iq]):+.4f}); gara Q0 - qff = {(ex0[iz] - ex0[iq]) / DELTA:+.2f} d "
          f"(eta cube {(exe[iz] - exe[iq]) / DELTA:+.2f}); max su Q calcolate a Q/qff = {out['q'][int(np.argmax(ex0))] / qff:.3f}")
    for tp, t in tabs:
        tq = np.asarray(t["q"], float)
        i0 = int(np.argmin(np.abs(np.asarray(t["omega"], float))))
        jz = int(np.argmin(np.abs(tq)))
        jq = int(np.argmin(np.abs(tq - qff)))
        print(f"   {tp}: g_c = {gc(float(t['ReInvGamma'][jq, i0])):+.4f}, gara = "
              f"{(float(t['ReInvGamma'][jz, i0]) - float(t['ReInvGamma'][jq, i0])) / DELTA:+.2f} d")
    if a.save:
        np.savez(a.save, **out)
        print(f"\nsalvato {a.save}")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="mode", required=True)
    f = sp.add_parser("free")
    f.add_argument("--P", type=float, default=0.9)
    r = sp.add_parser("run")
    r.add_argument("cubedir")
    r.add_argument("--table", action="append")
    r.add_argument("--lams", default="4:6:8:12:16")
    r.add_argument("--extra-q", default="")
    r.add_argument("--eta-alt", type=float, default=1e-3)
    r.add_argument("--save", default="")
    r.add_argument("--qfracs", default="0:0.02:0.05:0.1:0.2:0.5:0.9:0.97:1:1.02:1.05:1.2")
    r.add_argument("--per-decade", type=int, default=40)
    r.add_argument("--nper", type=int, default=20)
    r.add_argument("--dk", type=float, default=0.02)
    r.add_argument("--nwin", type=int, default=60)
    r.add_argument("--row-step", type=int, default=1)
    r.add_argument("--no-eta", action="store_true")
    a = ap.parse_args(argv)
    return cmd_free(a) if a.mode == "free" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
