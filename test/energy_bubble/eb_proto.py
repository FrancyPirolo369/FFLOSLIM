#!/usr/bin/env python3
"""PROTOTIPO INDIPENDENTE (non usato dalla produzione): ImPi(Q, Omega) a T = 0 con quadrature adattate.

    ImPi(Q, Omega) = (1/4pi) int deps K(eps, Omega) int_0^{Lambda^2} dx_up A_up(k_up, eps)
                                       int_0^pi dtheta A_dn(k_dn(theta), Omega - eps)
    x_up = k_up^2,  k_dn^2 = x_up + Q^2 - 2 k_up Q cos(theta)
    K = -1 su 0 < eps < Omega (particella-particella), +1 su Omega < eps < 0 (buca-buca)
(stessa normalizzazione di fflo/impi_table: pi/(2pi)^2 int p dp dphi; vuoto -1/8).

Niente reticoli e niente sottrazione di un modello QP: si integra il prodotto completo A_up A_dn, e in
ogni integrale 1D i nodi vanno dove sta la struttura:
  - A_sigma(k, w) = (1/pi) g / ((w - xi - (ReSigma - sigma0))^2 + g^2), g = |ImSigma| + eta, con Sigma
    interpolata bilinearmente dal cubo (righe k x w_base).  A frequenza fissa il picco in x = k^2 sta
    dove w - xi - ReSigma + sigma0 = 0: lo si trova (tutte le radici) e attorno si usa la mappa
    x = c + gamma tan(s) con Gauss-Legendre in s (una lorentziana diventa costante: esatta con pochi nodi);
  - altrove pannelli di Gauss-Legendre;
  - in theta lo stesso, con il picco del partner riportato in theta (alla tangenza cade sul bordo);
  - in eps pannelli GL addensati verso i due estremi della finestra (0 e Omega).

Uso (dalla radice di SLIM):
  python3 test/energy_bubble/eb_proto.py free            # fermioni liberi contro la formula esatta (arcoseno)
  python3 test/energy_bubble/eb_proto.py slice RUN       # cubi di una run: alcune fette (Q, Omega)
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAMBDA = 4.0
_GL = {}


def gl(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


# ---------------------------------------------------------------------------
# campo di Sigma e funzione spettrale
# ---------------------------------------------------------------------------
class Field:
    def __init__(self, mu, eta, k=None, wb=None, re=None, im=None, sigma0=0.0, mass=1.0):
        self.mu, self.eta, self.mass, self.sigma0 = float(mu), float(eta), float(mass), float(sigma0)
        self.k, self.wb, self.re, self.im = k, wb, re, im

    @classmethod
    def free(cls, mu, eta):
        return cls(mu, eta)

    @classmethod
    def from_cube(cls, path, spin):
        z = np.load(path, allow_pickle=True)
        k = np.asarray(z["k"], float)
        w = np.asarray(z["w"], float)
        wb = np.asarray(z["w_base"], float)
        re_rows, im_rows = np.asarray(z["ReS"], float), np.asarray(z["ImS"], float)
        re = np.empty((k.size, wb.size))
        im = np.empty_like(re)
        for i in range(k.size):
            wi = w[i] if w.ndim == 2 else w
            re[i] = np.interp(wb, wi, re_rows[i])
            im[i] = np.interp(wb, wi, im_rows[i])
        mu = float(z["mu_up"]) if spin == "up" else float(z["mu_dn"] if "mu_dn" in z.files else z["mu_down"])
        return cls(mu, float(z["eta"]), k, wb, re, im, float(z["sigma0"]), float(z["mass"]))

    def sigma(self, kq, wq):
        kq, wq = np.broadcast_arrays(np.asarray(kq, float), np.asarray(wq, float))
        if self.k is None:
            return np.zeros(kq.shape), np.zeros(kq.shape)
        k, wb = self.k, self.wb
        ik = np.clip(np.searchsorted(k, kq) - 1, 0, k.size - 2)
        tk = np.clip((kq - k[ik]) / (k[ik + 1] - k[ik]), 0.0, 1.0)
        iw = np.clip(np.searchsorted(wb, wq) - 1, 0, wb.size - 2)
        tw = np.clip((wq - wb[iw]) / (wb[iw + 1] - wb[iw]), 0.0, 1.0)
        out = []
        for f in (self.re, self.im):
            out.append((1 - tk) * (1 - tw) * f[ik, iw] + tk * (1 - tw) * f[ik + 1, iw]
                       + (1 - tk) * tw * f[ik, iw + 1] + tk * tw * f[ik + 1, iw + 1])
        return out[0], out[1]

    def denom(self, x, w):
        """D = w - xi - (ReSigma - sigma0) e g = |ImSigma| + eta, con x = k^2."""
        k = np.sqrt(np.maximum(x, 0.0))
        re, im = self.sigma(k, w)
        return w - (x / self.mass - self.mu) - (re - self.sigma0), np.abs(np.minimum(im, 0.0)) + self.eta

    def A(self, x, w):
        d, g = self.denom(x, w)
        return g / np.pi / (d * d + g * g)

    def peaks(self, w, x_lo, x_hi, n_scan=2400):
        """Picchi di A(x, w) in x = k^2 in [x_lo, x_hi]: radici di D(x) = 0 e larghezze g / |dD/dx|."""
        if x_hi <= x_lo:
            return []
        xs = np.linspace(x_lo, x_hi, n_scan)
        d, _ = self.denom(xs, w)
        out = []
        for j in np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
            x0, x1, d0, d1 = xs[j], xs[j + 1], d[j], d[j + 1]
            for _ in range(3):                      # secanti
                xm = x0 - d0 * (x1 - x0) / (d1 - d0)
                dm, _ = self.denom(np.array([xm]), w)
                dm = float(dm[0])
                if dm == 0.0:
                    break
                if np.sign(dm) == np.sign(d0):
                    x0, d0 = xm, dm
                else:
                    x1, d1 = xm, dm
            xm = x0 - d0 * (x1 - x0) / (d1 - d0)
            h = max(1e-7, 1e-4 * max(1.0, abs(xm)))
            dp, g = self.denom(np.array([xm - h, xm + h, xm]), w)
            slope = abs(dp[1] - dp[0]) / (2 * h)
            out.append((float(xm), float(g[2]) / max(slope, 1e-12)))
        return out


# ---------------------------------------------------------------------------
# quadrature 1D adattate
# ---------------------------------------------------------------------------
def adapt_rule(L, R, peaks, breaks=(), n_pk=16, n_reg=6, h_max=0.5, span=40.0):
    """Nodi e pesi su [L, R]: mappa x = c + gamma tan(s) attorno a ogni picco (c, gamma), GL altrove."""
    if R <= L:
        return np.zeros(0), np.zeros(0)
    iv = []
    for c, g in sorted(peaks):
        if g <= 0 or not (L - span * g < c < R + span * g):
            continue
        lo, hi = max(L, c - span * g), min(R, c + span * g)
        if hi > lo:
            iv.append([lo, hi, c, g])
    for i in range(len(iv) - 1):                       # niente sovrapposizioni: taglio a meta' fra i centri
        if iv[i][1] > iv[i + 1][0]:
            m = 0.5 * (iv[i][2] + iv[i + 1][2])
            iv[i][1] = max(iv[i][0], min(iv[i][1], m))
            iv[i + 1][0] = min(iv[i + 1][1], max(iv[i + 1][0], m))
    nodes, weights = [], []
    xg, wg = gl(n_pk)
    for lo, hi, c, g in iv:
        for a, b in ((lo, min(c, hi)), (max(c, lo), hi)):
            if b - a <= 0:
                continue
            sa, sb = math.atan((a - c) / g), math.atan((b - c) / g)
            s = 0.5 * (sb - sa) * xg + 0.5 * (sb + sa)
            nodes.append(c + g * np.tan(s))
            weights.append(0.5 * (sb - sa) * wg * g / np.cos(s) ** 2)
    # tratti liberi: pannelli che crescono geometricamente a partire dalla larghezza del pannello di picco
    # adiacente (la coda 1/x^2 della lorentziana), poi spezzati ai punti di taglio e a h_max
    free, cur, h_cur = [], L, None
    for lo, hi, _, g in sorted(iv):
        if hi <= lo:
            continue
        if lo > cur:
            free.append((cur, lo, h_cur, span * g))
        cur, h_cur = max(cur, hi), span * g
    if cur < R:
        free.append((cur, R, h_cur, None))
    xr, wr = gl(n_reg)
    for a, b, ha, hb in free:
        pts = {a, b} | {x for x in breaks if a < x < b}
        for start, h0, sgn in ((a, ha, +1), (b, hb, -1)):
            if h0 is None:
                continue
            d = h0
            while d < (b - a):
                pts.add(start + sgn * d)
                d *= 2.5
        pts = sorted(pts)
        for u, v in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil((v - u) / h_max)))
            e = np.linspace(u, v, n + 1)
            for uu, vv in zip(e[:-1], e[1:]):
                nodes.append(0.5 * (vv - uu) * xr + 0.5 * (vv + uu))
                weights.append(0.5 * (vv - uu) * wr)
    return np.concatenate(nodes), np.concatenate(weights)


def feasible(e, Q, om, up, dn, xmax):
    """Firma geometrica a eps fissato: per ogni coppia (picco su, picco partner) se il picco del partner
    cade dentro [(k_up - Q)^2, (k_up + Q)^2].  Dove cambia, l'integrando in eps ha una singolarita' 1/sqrt."""
    pu = up.peaks(e, 0.0, xmax, n_scan=800)
    pd = dn.peaks(om - e, 0.0, (math.sqrt(xmax) + Q) ** 2, n_scan=800)
    sig = []
    for xu, _ in pu:
        ku = math.sqrt(max(xu, 0.0))
        for xd, _ in pd:
            sig.append(((ku - Q) ** 2 < xd < (ku + Q) ** 2))
            sig.append(xd > xu)            # gusci coincidenti: a Q -> 0 la finestra geometrica si chiude li'
    return (len(pu), len(pd), tuple(sig))


def eps_breaks(Q, om, up, dn, xmax, n_scan=300):
    a, b = sorted((0.0, om))
    es = np.linspace(a, b, n_scan + 1)[1:-1]
    sigs = [feasible(e, Q, om, up, dn, xmax) for e in es]
    out = []
    for j in range(len(es) - 1):
        if sigs[j] != sigs[j + 1]:
            lo, hi, slo = es[j], es[j + 1], sigs[j]
            for _ in range(12):                     # bisezione sul cambio di firma
                m = 0.5 * (lo + hi)
                if feasible(m, Q, om, up, dn, xmax) == slo:
                    lo = m
                else:
                    hi = m
            out.append(0.5 * (lo + hi))
    return out


def graded_panels(a, b, brk, h0=1e-5, ratio=2.5):
    """Estremi di pannelli in [a, b] che si stringono geometricamente verso ogni punto di brk (e verso a, b)."""
    pts = {a, b}
    for c in list(brk) + [a, b]:
        if not (a <= c <= b):
            continue
        pts.add(c)
        d = h0
        while d < (b - a):
            for x in (c - d, c + d):
                if a < x < b:
                    pts.add(x)
            d *= ratio
    pts = np.array(sorted(pts))
    return pts[np.concatenate(([True], np.diff(pts) > 1e-14))]


def eps_rule(om, n=4, brk=()):
    """Finestra [0, Omega] (o [Omega, 0]) in pannelli GL graduati verso gli estremi e i punti singolari."""
    a, b = sorted((0.0, om))
    pts = graded_panels(a, b, brk)
    xg, wg = gl(n)
    nodes = np.concatenate([0.5 * (v - u) * xg + 0.5 * (v + u) for u, v in zip(pts[:-1], pts[1:])])
    weights = np.concatenate([0.5 * (v - u) * wg for u, v in zip(pts[:-1], pts[1:])])
    return nodes, weights


# ---------------------------------------------------------------------------
# la bolla
# ---------------------------------------------------------------------------
def impi(Q, om, up, dn, lam=LAMBDA, n_eps=4, n_pk=16, n_reg=6, n_th_reg=6):
    if om == 0.0:
        return 0.0
    Q = max(float(Q), 1e-6)               # Q = 0 come limite Q -> 0+ (ReGamma^-1 e' pari in Q)
    K = -1.0 if om > 0 else +1.0
    xmax = lam * lam
    brk = eps_breaks(Q, om, up, dn, xmax)
    ens, ews = eps_rule(om, n_eps, brk)
    total = 0.0
    xr, wr = gl(n_th_reg)
    for e, we in zip(ens, ews):
        nu = om - e
        pk_up = up.peaks(e, 0.0, xmax)
        xu, wu = adapt_rule(0.0, xmax, pk_up, breaks=(up.mu,), n_pk=n_pk, n_reg=n_reg)
        a_up = up.A(xu, e)
        keep = a_up * wu > 1e-14 * max(1e-300, float(np.max(np.abs(a_up * wu))))
        xu, wu, a_up = xu[keep], wu[keep], a_up[keep]
        ku = np.sqrt(np.maximum(xu, 0.0))
        if Q <= 1e-12:
            inner = np.pi * dn.A(xu, nu)
        else:
            pk_dn = dn.peaks(nu, 0.0, (math.sqrt(xmax) + Q) ** 2)
            inner = np.empty(xu.size)
            for i, (kv, xv) in enumerate(zip(ku, xu)):
                lo, hi = (kv - Q) ** 2, (kv + Q) ** 2
                pks = []
                for c, g in pk_dn:
                    if lo - 40 * g < c < hi + 40 * g:
                        cc = min(max(c, lo), hi)
                        th = math.acos(min(1.0, max(-1.0, (lo + hi - 2 * cc) / (hi - lo))))
                        t1 = math.acos(min(1.0, max(-1.0, (lo + hi - 2 * min(hi, c + g)) / (hi - lo))))
                        t0 = math.acos(min(1.0, max(-1.0, (lo + hi - 2 * max(lo, c - g)) / (hi - lo))))
                        gth = max(0.5 * (t1 - t0), 1e-9)
                        pks.append((th, gth))
                th, wt = adapt_rule(0.0, math.pi, pks, n_pk=n_pk, n_reg=n_th_reg, h_max=0.4)
                xd = 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(th)
                inner[i] = float(np.sum(wt * dn.A(xd, nu)))
        total += we * K * float(np.sum(wu * a_up * inner))
    return total / (4.0 * np.pi)


# ---------------------------------------------------------------------------
def impi_exact_free(Q, w, c1, c2):
    """Fermioni liberi, Gamma -> 0: formula all'arcoseno (stessa di test/qpqp_ab_probe.py)."""
    d0 = c1 - c2 - w + Q * Q
    A, B, C = -4.0, 4.0 * Q * Q - 4.0 * d0, 4.0 * Q * Q * c1 - d0 * d0
    disc = B * B - 4 * A * C
    if disc <= 0:
        return 0.0
    r1, r2 = sorted(((-B + math.sqrt(disc)) / (2 * A), (-B - math.sqrt(disc)) / (2 * A)))
    lo, hi = max(min(0.0, w), r1, -c1), min(max(0.0, w), r2, w + c2)
    if hi <= lo:
        return 0.0
    f = lambda e: math.asin(min(1.0, max(-1.0, (2 * e - r1 - r2) / (r2 - r1))))
    return -math.copysign(1.0, w) / (4 * math.pi) * 0.5 * (f(hi) - f(lo))


def run_free(a):
    print("fermioni liberi, eta = %g: ImPi prototipo contro formula esatta (Gamma -> 0)" % a.eta)
    for P in (0.5, 0.75, 0.9):
        mu_u, mu_d = 1 + P, 1 - P
        qff = math.sqrt(mu_u) - math.sqrt(mu_d)
        up, dn = Field.free(mu_u, a.eta), Field.free(mu_d, a.eta)
        for qr in (0.0, 0.5, 0.97, 1.0, 1.03, 1.5):
            row = []
            for om in (-0.4, -0.05, 0.01, 0.3, 2.0):
                t0 = time.time()
                v = impi(qr * qff, om, up, dn)
                ex = impi_exact_free(qr * qff, om, mu_u, mu_d)
                row.append(f"Om={om:+.2f}: {v:+.6f}/{ex:+.6f}")
            print(f"  P={P} Q={qr:4.2f}qff  " + "  ".join(row), flush=True)


def run_slice(a):
    base = os.path.join(HERE, "out", "cluster", a.run)
    upp = sorted(glob.glob(os.path.join(base, "iter*", "next_cubes", "A_komega_spinup_iter*.npz")))[-1]
    up, dn = Field.from_cube(upp, "up"), Field.from_cube(upp.replace("spinup", "spindown"), "down")
    qff = math.sqrt(up.mu) - math.sqrt(dn.mu)
    print(f"{a.run}: mu_up={up.mu}, mu_dn={dn.mu}, qff={qff:.4f}, sigma0 up/dn = {up.sigma0:+.4f}/{dn.sigma0:+.4f}")
    for qr, om in ((1.0, 0.3), (1.0, -0.05), (0.014, 0.3), (0.0, 0.3), (1.0, 1.0), (0.5, -0.3)):
        res = []
        for n_pk, n_reg in ((8, 4), (16, 6), (24, 8)):
            t0 = time.time()
            v = impi(qr * qff, om, up, dn, n_pk=n_pk, n_reg=n_reg, n_th_reg=n_reg, n_eps=n_reg - 2)
            res.append(f"({n_pk},{n_reg}): {v:+.6f} [{time.time() - t0:.1f}s]")
        print(f"  Q={qr:5.3f}qff Om={om:+5.2f}  " + "  ".join(res), flush=True)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("free", "slice"))
    ap.add_argument("run", nargs="?", default="P0p75_gmax")
    ap.add_argument("--eta", type=float, default=1e-5)
    a = ap.parse_args(argv)
    return run_free(a) if a.mode == "free" else run_slice(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
