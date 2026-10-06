#!/usr/bin/env python3
"""Errore di quadratura del RIFERIMENTO analitico nella KK del residuo di pair_gamma, a Omega = 0.

In proxy_residual: ReG^-1 = ReG^-1_ref(esatto) - KK_num[(ImPi_num - ImPi_ref) w], con w il taper.  ImPi_ref
e' la T-matrice libera analitica a omega + i eta (eta = 0.002): bordi di Pauli larghi ~eta, che la griglia
Omega della tabella non risolve.  Qui, per ogni riga Q:
    e(Q) = (1/pi) int ImPi_ref w / Omega  [griglia nativa]  -  la stessa  [griglia fine, Pi_ref analitico]
con la stessa quadratura lineare a tratti (esatta per f lineare: int f/x su ogni cella in forma chiusa).
ReG^-1 corretto = ReG^-1 - e(Q): ReG^-1 = ReG^-1_ref - KK[(ImPi_num - ImPi_ref) w] contiene +KK[ImPi_ref w],
calcolata dalla produzione sulla griglia nativa; quella giusta e' sulla griglia fine.

Uso: python3 test/integration_audit/ref_kk_probe.py PAIR_TABLE IMPI_TABLE QFF
"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
from fflo.kramers_kronig import proxy_full_pi_from_gamma  # noqa: E402
from fflo.pair_gamma import _taper_weight_on_grid, _taper_breakpoints, load_im_source  # noqa: E402


def kk_at_zero(x, f):
    """PV (1/pi) int f(x)/x dx con f lineare a tratti sui nodi x: si sottrae f(0) (g = f - f(0) si annulla
    in 0, quindi g/x e' regolare e integrabile cella per cella in forma chiusa) e si aggiunge f(0) ln(b/|a|)
    sull'intervallo intero."""
    x = np.asarray(x, float); f = np.asarray(f, float)
    f0 = float(np.interp(0.0, x, f))
    g = f - f0
    a, b = x[:-1], x[1:]
    ga, gb = g[:-1], g[1:]
    s = (gb - ga) / (b - a)
    same = ((a > 0) & (b > 0)) | ((a < 0) & (b < 0))
    val = np.where(same, (ga - s * a) * np.log(np.abs(np.where(same, b / np.where(a != 0, a, 1), 1.0))) + s * (b - a),
                   s * (b - a))                  # cella che tocca o contiene 0: g = s x esattamente
    return (float(np.sum(val)) + f0 * np.log(x[-1] / abs(x[0]))) / np.pi


def main(pair_path, impi_path, qff, taper_mode="qkin_cosine", t0=0.0, t1=20.0, eta=0.002, step=2.5e-4):
    pg = np.load(pair_path, allow_pickle=True)
    q, w = np.asarray(pg["q"], float), np.asarray(pg["omega"], float)
    _, _, _, _, eps0, mu_up, mu_dn, p_cut, _, _ = load_im_source(impi_path, "auto")
    i0 = int(np.argmin(np.abs(w)))
    re0 = np.asarray(pg["ReInvGamma"], float)[:, i0]
    span = abs(mu_up - mu_dn) + 0.5
    errs = np.zeros(q.size)
    for i, qv in enumerate(q):
        br = [x for x in _taper_breakpoints(float(qv), mode=taper_mode, start=t0, stop=t1, mu_up=mu_up, mu_dn=mu_dn)
              if w[0] < x < w[-1]]
        xn = np.unique(np.concatenate([w, br]))
        xf = np.unique(np.concatenate([xn, np.arange(-span, span + step / 2, step)]))
        out = []
        for x in (xn, xf):
            ref = np.imag(proxy_full_pi_from_gamma(x, q=float(qv), eps0=eps0, mu_up=mu_up, mu_dn=mu_dn,
                                                   eta_gamma=eta, cutoff=p_cut))
            wt = _taper_weight_on_grid(float(qv), x, mode=taper_mode, start=t0, stop=t1, mu_up=mu_up, mu_dn=mu_dn)
            if x is xn:
                ref = np.interp(xn, w, np.asarray(pg["ImPiReference"], float)[i])   # quello che la tabella ha usato
            out.append(kk_at_zero(x, ref * wt))
        errs[i] = out[0] - out[1]
    return q, re0, errs


if __name__ == "__main__":
    pair_path, impi_path, qff = sys.argv[1], sys.argv[2], float(sys.argv[3])
    q, re0, e = main(pair_path, impi_path, qff)
    iq = int(np.argmin(np.abs(q - qff)))
    d0 = (re0 - re0[iq]) / 1e-3
    d1 = ((re0 - e) - (re0[iq] - e[iq])) / 1e-3
    sel = (q > 1e-9) & (q <= 0.35 * qff)
    np.savez(os.path.join(os.path.dirname(pair_path), "ref_kk_correction.npz"), q=q, re0=re0, e=e)
    print("Q/qff   ReG^-1-qff [delta]: tabella | errore riferimento e(Q) | corretto")
    for j in np.flatnonzero(sel | (np.abs(q / qff - 1) < 0.06)):
        print(f"  {q[j]/qff:.3f}   {d0[j]:+7.2f}   {(e[j]-e[iq])/1e-3:+7.2f}   {d1[j]:+7.2f}")
    r2 = lambda a: np.sqrt(np.mean(np.diff(a[sel], 2) ** 2))
    print(f"rugosita' Q <= 0.35 qff (RMS 2a diff): tabella {r2(d0):.3f}  corretto {r2(d1):.3f} delta")
