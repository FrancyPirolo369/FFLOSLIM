#!/usr/bin/env python3
"""Diagramma di fase: g_c in ascissa, P in ordinata, risultato nuovo contro FFLO30 (luglio).

Per ogni P prende l'ultima iterazione della prima famiglia disponibile, nell'ordine di
--families (default prod_union, prodg, gmax, prod, x75a0p3, x75a0p6: prima PROD_UNION, poi il massimo globale, poi PROD
a qff, poi le prove ad alta P di submit_highP_x.sh ripartite dalla 0.75, alpha 0.3 prima).
g_c = -4 pi shift - ln2/2, lo shift e' quello applicato da density (riga PAIR di loop.log).
Per le famiglie pinnate a qff (prod, etaexact) --gmax-corr aggiunge la correzione al primo
ordine per pinnare il massimo globale di ReGamma^-1(Q,0) (dall'ultima snapshot).
Ogni punto e' classificato dal Q del Thouless all'ultima iterazione (Q= della riga PAIR;
per le pinnate a qff con la correzione, il Q del massimo globale): Q < --pbcs-qmax * qff
e' pBCS, altrimenti FFLO.  Una spline di smoothing g(P) per ramo, di colore diverso
(--smooth = scarto tipico atteso per punto; 0 = interpolante); forma del marker =
ramo, colore = ramo (la famiglia sta nella tabella stampata).  --exclude toglie delle P.

Campo medio (T = 0, soglia FFLO): P_c = eps0 sqrt(2/eps0 - 1) = sqrt(eps0 (2 - eps0)), con
g = ln(eps0/2)/2 (la stessa relazione eps0 <-> g dei punti numerici), eps0 in (0, 1].

NSCT (T-matrix non autoconsistente): test/data/nsct_critical_line.txt, due rami (debole,
P <= 0.1 per g <= -0.8; forte, da g = 0.2 fino a P ~ 1).  Densita' fissata: la sua
normalizzazione di eps0 coincide con mu_avg = 1 solo ad accoppiamento debole.

FFLO30: r30b_pull/critical_line_status.txt (scansione r30 0.05-0.40 e continuazioni r30b
0.35-0.70, 9-12 iterazioni; stessa formula di g_c, eps0 = 1).  Il punto a P = 0.70 ha
ReGamma^-1(qff) < 0 (canale q ~ 0 gia' vincente): escluso salvo --fflo30-all.

Uso (dalla radice di SLIM):
  python3 test/plot_phase_diagram.py [--exclude 0.5] [--scan] [--out out/cluster/plots/phase_diagram.png]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e3df"
C_NEW, C_F30, C_SCAN, C_MF, C_NSCT = "#eb6834", "#2a78d6", "#6b6b68", "#1baf7a", "#e87ba4"
C_PBCS = "#7b4fc4"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.5,
})


def g_of(shift):
    return -4.0 * np.pi * shift - 0.5 * np.log(2.0)


def last_shift(loop_log):
    """(iterazione, g_c, deriva di g_c per iterazione sulle ultime 3, Q/qff selezionato).

    Il Q selezionato e' quello della riga PAIR (Q=...); le run pinnate a qff con il
    motore vecchio non lo scrivono e restituiscono None."""
    its, gs, qs, qff = [], [], [], None
    for line in open(loop_log):
        m = re.search(r"qff=([\d.]+)", line)
        if m and qff is None:
            qff = float(m[1])
        m = re.search(r"iter (\d+) PAIR: .*shift=([+-][\d.eE+-]+)(?:.*Q=([\d.]+))?", line)
        if m:
            its.append(int(m[1]))
            gs.append(g_of(float(m[2])))
            qs.append(float(m[3]) if m[3] else None)
    if not its:
        return None
    n = len(gs)
    drift = (gs[-1] - gs[max(0, n - 3)]) / max(min(2, n - 1), 1) if n > 1 else np.nan
    qrel = qs[-1] / qff if (qs[-1] is not None and qff) else None
    return its[-1], gs[-1], drift, qrel


def gmax_correction(run_dir):
    """Correzione di g_c al massimo globale di ReGamma^-1(Q,0) e Q/qff di quel massimo."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "snap", "iter*.npz")))
    if not snaps:
        return 0.0, None
    z = np.load(snaps[-1])
    q, w = np.asarray(z["q"]), np.asarray(z["omega"])
    re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    qff = float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    imax = int(np.nanargmax(re0))
    return float(-4.0 * np.pi * (re0[imax] - re0[iq])), float(q[imax] / qff)


def two_candidate(run_dir):
    """g_c col pin a due candidati 2D (qff esatto o la piu' piccola Q > 0, come density
    --thouless-q-mode qff-or-zero) dall'ultima snapshot: (g_c, Q/qff del candidato)."""
    snaps = sorted(glob.glob(os.path.join(run_dir, "snap", "iter*.npz")))
    if not snaps:
        return None
    z = np.load(snaps[-1])
    q, w = np.asarray(z["q"], float), np.asarray(z["omega"], float)
    re0 = np.asarray(z["ReInvGamma"], float)[:, int(np.argmin(np.abs(w)))]
    qff = float(z["qff"])
    iq = int(np.argmin(np.abs(q - qff)))
    nz = np.flatnonzero(q > 1.0e-9)
    i0 = int(nz[np.argmin(q[nz])])
    j = iq if re0[iq] >= re0[i0] else i0
    return float(g_of(re0[j])), float(q[j] / qff)


def collect(base, families, gmax_corr, min_iters, pbcs_qmax=0.1, two_cand=False):
    """Ultima iterazione per P; phase = pBCS se il Q del Thouless e' < pbcs_qmax * qff."""
    pts = {}
    for fam in families:
        for d in sorted(glob.glob(os.path.join(base, f"P0p*_{fam}"))):
            m = re.search(r"P0p(\d+)_", os.path.basename(d) + "_")
            p = float("0." + m.group(1))
            if p in pts or not os.path.exists(os.path.join(d, "loop.log")):
                continue
            r = last_shift(os.path.join(d, "loop.log"))
            if r is None or r[0] < min_iters:
                continue
            corr, qrel = 0.0, r[3]
            tc = two_candidate(d) if two_cand else None
            if tc is not None:
                corr, qrel = tc[0] - r[1], tc[1]
            elif gmax_corr and fam in ("prod", "etaexact"):
                # pinnate a qff: g e Q diventano quelli del massimo globale
                corr, qrel = gmax_correction(d)
            if qrel is None:
                qrel = 1.0                      # pinnata a qff senza correzione
            phase = "pBCS" if qrel < pbcs_qmax else "FFLO"
            pts[p] = dict(p=p, fam=fam, it=r[0], g=r[1] + corr, drift=r[2], corr=corr,
                          qrel=qrel, phase=phase)
    return pts


def read_fflo30(path, keep_all):
    out = {}
    for line in open(path):
        f = line.split()
        if not f or f[0] == "series":
            continue
        p, reinv, gc = float(f[1]), float(f[4]), float(f[5])
        if reinv < 0 and not keep_all:
            continue
        out[p] = gc if f[0].startswith("r30b") or p not in out else out[p]   # r30b prevale su r30
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.path.join(HERE, "out", "cluster"))
    ap.add_argument("--families", default="prod_union_fine,prod_union,prodg,gmax,prod,x75a0p3,x75a0p6")
    ap.add_argument("--min-iters", type=int, default=1)
    ap.add_argument("--no-gmax-corr", action="store_true")
    ap.add_argument("--two-cand", action="store_true",
                    help="g_c col pin a due candidati (qff esatto o Q -> 0) ricalcolato dall'ultima "
                         "snapshot; sostituisce la correzione al massimo globale (ritirata in 2D)")
    ap.add_argument("--xerr", default="",
                    help="barre d'errore orizzontali su g_c: 'e' per tutti o 'P:e,P:e,...'")
    ap.add_argument("--exclude", default="", help="P da togliere, separate da virgole ('' = nessuna)")
    ap.add_argument("--smooth", type=float, default=0.03,
                    help="scarto tipico per punto della spline in g_c (0 = interpolante)")
    ap.add_argument("--pbcs-qmax", type=float, default=0.1,
                    help="Q del Thouless sotto questa frazione di qff = pBCS (Q = 0), sopra = FFLO")
    ap.add_argument("--scan", action="store_true", help="aggiunge la scansione SLIM (etaexact, it 30)")
    ap.add_argument("--fflo30", default=os.path.join(os.path.dirname(HERE), "r30b_pull",
                                                     "critical_line_status.txt"))
    ap.add_argument("--fflo30-all", action="store_true")
    ap.add_argument("--no-mean-field", action="store_true")
    ap.add_argument("--nsct", default=os.path.join(HERE, "test", "data", "nsct_critical_line.txt"),
                    help="file NSCT ('' = non disegnarlo)")
    ap.add_argument("--xlim", default="-3.6,2.0")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "cluster", "plots", "phase_diagram.png"))
    a = ap.parse_args(argv)
    from scipy.interpolate import UnivariateSpline

    excl = {round(float(x), 4) for x in a.exclude.split(",") if x.strip()}
    pts = collect(a.base, [f.strip() for f in a.families.split(",")],
                  not a.no_gmax_corr and not a.two_cand, a.min_iters, a.pbcs_qmax, a.two_cand)
    xerr = {}
    if a.xerr:
        if ":" in a.xerr:
            xerr = {round(float(k), 4): float(v) for k, v in (x.split(":") for x in a.xerr.split(","))}
        else:
            xerr = {round(p, 4): float(a.xerr) for p in pts}
    keep = [pts[p] for p in sorted(pts) if round(p, 4) not in excl]
    print(f"{'P':>5s}  famiglia  it   g_c     ({'corr. 2 candidati' if a.two_cand else 'corr. max globale'})  deriva/it  Q/qff  ramo")
    for p in sorted(pts):
        r = pts[p]
        tag = "  ESCLUSO" if round(p, 4) in excl else ""
        print(f"{p:5.2f}  {r['fam']:8s} {r['it']:3d}  {r['g']:+.3f}   ({r['corr']:+.3f})          "
              f"{r['drift']:+.4f}    {r['qrel']:5.3f}  {r['phase']}{tag}")

    fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
    if not a.no_mean_field:
        eps0 = np.geomspace(1e-4, 1.0, 600)
        ax.plot(0.5 * np.log(eps0 / 2.0), eps0 * np.sqrt(2.0 / eps0 - 1.0), "-", color=C_MF,
                lw=1.4, label="mean-field/psct")
    if a.nsct and os.path.exists(a.nsct):
        d = np.loadtxt(a.nsct)
        d = d[np.argsort(d[:, 2])]
        cut = np.flatnonzero(np.diff(d[:, 2]) > 0.5) + 1          # rami separati da un buco in g
        for i, br in enumerate(np.split(d, cut)):
            ax.plot(br[:, 2], br[:, 3], "-.", color=C_NSCT, lw=1.5, label="NSCT" if i == 0 else None)
    if os.path.exists(a.fflo30):
        f30 = read_fflo30(a.fflo30, a.fflo30_all)
        ps = sorted(f30)
        ax.plot([f30[p] for p in ps], ps, "o--", color=C_F30, ms=4.5, lw=1.2,
                label="luglio")
    if a.scan:
        sc = collect(a.base, ["etaexact"], False, 1)
        ps = [p for p in sorted(sc) if p <= 0.7]
        ax.plot([sc[p]["g"] for p in ps], ps, "o:", color=C_SCAN, ms=4, lw=1.0,
                label="slim scan (start of last week)")
    # una spline per ramo, scelto dal Q del Thouless: FFLO (Q != 0) e pBCS (Q = 0)
    col_of = {"FFLO": C_NEW, "pBCS": C_PBCS}
    for name, lab in (("FFLO", "spline FFLO (Q ≠ 0)"), ("pBCS", "spline pBCS (Q = 0)")):
        rr = [r for r in keep if r["phase"] == name]
        if len(rr) < 2:
            continue
        P = np.array([r["p"] for r in rr]); G = np.array([r["g"] for r in rr])
        k = min(3, len(P) - 1)                                    # 2 punti = retta, 3 = parabola
        spl = UnivariateSpline(P, G, k=k, s=len(P) * a.smooth ** 2)
        pf = np.linspace(P.min(), P.max(), 300)
        ax.plot(spl(pf), pf, "-", color=col_of[name], lw=2.2, label=lab)
        res = G - spl(P)
        print(f"scarti dalla spline {name}: " + "  ".join(f"{p:.2f}:{x:+.3f}" for p, x in zip(P, res)))
    # stesso marker e colore per tutto il ramo, qualunque sia la famiglia della run
    mark_of = {"FFLO": "s", "pBCS": "^"}
    for phase, lab in (("FFLO", "FFLO (Q ≠ 0)"), ("pBCS", "pBCS (Q = 0)")):
        rr = [r for r in keep if r["phase"] == phase]
        if rr:
            ex = [xerr.get(round(r["p"], 4), 0.0) for r in rr]
            if any(ex):
                ax.errorbar([r["g"] for r in rr], [r["p"] for r in rr], xerr=ex, fmt="none",
                            ecolor=col_of[phase], elinewidth=1.0, capsize=2.5, alpha=0.8)
            ax.plot([r["g"] for r in rr], [r["p"] for r in rr], mark_of[phase],
                    color=col_of[phase], ms=7, mec="#fcfcfb", mew=1.2, label=lab)
    for r in keep:
        ax.annotate(f"it {r['it']}", (r["g"], r["p"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=7, color=MUTED)
    ymax = max([r["p"] for r in keep] + [0.7]) + 0.08
    x0, x1 = (float(v) for v in a.xlim.split(","))
    ax.set(xlabel="g_c", ylabel="P", ylim=(0, ymax), xlim=(x0, x1),)
    ax.set_ylim(0, 1)
    ax.set_xlim(x0, 5)
    ax.legend(fontsize=8, loc="upper left")
    if excl:
        ax.text(0.99, 0.01, "esclusi: P = " + ", ".join(f"{x:g}" for x in sorted(excl)),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("scritto", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
