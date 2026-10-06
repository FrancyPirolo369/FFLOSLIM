#!/usr/bin/env python3
"""TEST DRIVER 2026-09-24 -- copy of ../run_fflo.py with the knobs the cluster test
matrix needs (see test/submit.sh).  The core pipeline (fflo/, run_fflo.py) is not
touched; this file only adds:
  --omega-mode      pair omega features (default zero_and_thresholds, not zero)
  --bubble-lambda   Lambda of the pair bubble, also in the q_table_max budget
  --k-update-max    explicit Sigma update window (+ --allow-partial-q-support)
  --pair-shift-freeze   freeze the Thouless shift at this run's first iteration
  --time-budget     never start an iteration that would overrun the walltime
  an NK4dn log line per iteration (n(k) k^4 of the minority spin)

The FFLO self-consistency loop: one entry point, one place for every default.

Replaces submit_long.sh + run_long.sh + loop_wide.py.

WHY
---
In the old tree a run's configuration did not live in any single file.  It was
assembled from five layers -- Python defaults, loop_wide.py's LOOP_* defaults,
run_long.sh's exports and inline assignments, submit_long.sh's sbatch --export,
and whatever happened to be in the submitting shell (because of --export=ALL).
Six of the twelve loop knobs were assigned INLINE in run_long.sh, so exporting
them had no effect and failed silently.  On 2026-09-21 an identical re-run of
gammafix/iter019 produced an ImSigma differing by 18% at large k and the cause
could not be identified, because none of the governing variables was written down.

Here every knob is a field of CONFIG below, with its default and a one-line
rationale.  Nothing is read from the environment behind your back, and every run
writes the full resolved configuration to <out>/config.json before doing any work.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

# the repo root (this file lives in test/): cwd of the stages and base of seed-dir
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# EVERY KNOB, ONE PLACE.  (name, default, help)
# ---------------------------------------------------------------------------
CONFIG = [
    # --- what to run -------------------------------------------------------
    ("out",          "out/run",  "output directory; iterNNN/ are created under it"),
    ("iters",        20,         "how many iterations to run in this invocation"),
    ("target",       0,          "stop once this many iterations exist in total "
                                 "(0 = just do --iters more)"),
    ("up",           "",         "starting spin-up cube; empty = resume, else seed"),
    ("down",         "",         "starting spin-down cube"),
    ("seed-dir",     "seeds",    "where the warm seed cubes live"),
    ("resume",       True,       "continue from the last iteration that has BOTH "
                                 "next cubes; a half-written iteration is discarded"),
    ("workers",      8,          "process pool size for both stages"),

    # --- the self-consistency itself --------------------------------------
    ("alpha",        0.3,        "Picard mixing: Sigma_new = a*fresh + (1-a)*seed"),
    ("delta",        1e-3,       "subcritical delta: Re Gamma^-1(qff,0) is pinned "
                                 "to -delta"),
    ("pair-shift-fixed", None,   "freeze the Thouless shift at this value instead of "
                                 "recomputing it every iteration.  The shift "
                                 "constrains ONE point while the contact is an "
                                 "integral over (Q,Omega); re-pinning is the prime "
                                 "suspect for the runaway (contact 0.576 -> 1.081 in "
                                 "13 iterations, vs 0.2745 converged when frozen)"),
    ("pair-shift-freeze", False, "freeze the Thouless shift at the value measured by "
                                 "THIS run's first iteration (read back from "
                                 "iter<first>/density on resume).  Use this instead "
                                 "of --pair-shift-fixed: the -0.019825 used so far is "
                                 "the turbo-grid value, 34% off the converged one"),

    # --- the budget --------------------------------------------------------
    ("pintmax",      6.0,        "SIGMA_PN_PINTMAX: internal momentum cutoff in "
                                 "Sigma.  NOT an accuracy knob -- 89% of the pair "
                                 "weight sits at Q<=2, so the dominant internal "
                                 "momentum is p~k and this cancels the dominant "
                                 "channel for k > pintmax+1"),
    ("q-cap",        20.0,       "ceiling on q_table_max; the real limit is "
                                 "k_max_cube - bubble_lambda - 0.1"),
    ("bubble-lambda", 4.0,       "Lambda of the pair bubble (pairbuild --bubble-p-int-max). "
                                 "It cuts |k_up| < Lambda, NOT the relative momentum, "
                                 "while the vacuum reference is Galilean: for Q >~ 2 "
                                 "Lambda the whole vacuum ImPi (1/8) is missing and "
                                 "Gamma's bound state is pushed to threshold (measured "
                                 "2026-09-24).  Raising it shrinks q_table_max"),
    ("truncation-fix", "none",   "none|physical|label: add to ImPi the free weight "
                                 "outside the disk |k_up| < bubble_lambda before the KK "
                                 "(test/fix_truncation.py).  physical = energies "
                                 "k^2/m - (mu + sigma0), what the cube has at large k: "
                                 "the validated choice (2026-09-24 Lambda scan)"),
    ("taper-mode",   "qkin_cosine", "pair_gamma --residual-omega-taper-mode "
                                 "(cosine|hard|none|qkin_cosine|qkin_hard|qcut_cosine; qcut = "
                                 "residuo intero fino a dove la bolla con |p| <= Lambda e' completa, "
                                 "sfumato solo negli ultimi taper-stop, 2026-10-04).  Any "
                                 "value other than the pairbuild one runs pair_gamma "
                                 "separately, like --truncation-fix"),
    ("taper-stop",   20.0,       "pair_gamma --residual-omega-taper-stop"),
    ("im-sign-guard", "on",      "pair_gamma --im-inv-sign-guard on|off (clamps "
                                 "wrong-sign Im Gamma^-1; 7% of the cells at P=0.65)"),
    ("p-nodes",      0,          "pairbuild --p-nodes (0 = profile default, 31 for "
                                 "turbo on [0, Lambda]: scale it with Lambda or a larger "
                                 "Lambda also means a coarser p grid)"),
    ("coherent-nk",  0,          "pairbuild --coherent-nk (0 = profile default, 96 "
                                 "for turbo on [0, Lambda]; scale it with Lambda too)"),
    ("k-update-max", 0.0,        "override k_update_max (0 = q_table_max - pintmax).  "
                                 "The tail channel (pair at Q<~3, fermion at p~k) "
                                 "needs pintmax >= k_update_max + ~3; the formula does "
                                 "not enforce it"),

    # --- resolution --------------------------------------------------------
    ("core-n",       40,         "q nodes in [0, 2.2], packed around 0 and qff"),
    ("tail-n",       19,         "q nodes above 2.2 (auto-rescaled with the span)"),
    ("sigma-nk",     24,         "k points where Sigma is really evaluated"),
    ("sigma-nomega", 41,         "omega points where Sigma is really evaluated; "
                                 "cost is sigma_nk * sigma_nomega per spin"),
    ("n-theta",      32,         "angular nodes in the density stage"),
    ("omega-chunk",  1000,       "density --omega-chunk: Sigma omega nodes per call of the row engine.  "
                                 "The theta-averaged fermion table (1200 Q x 32 theta x ~1340 eps) does "
                                 "not depend on omega but is rebuilt at every call: density's own default "
                                 "16 rebuilds it 7x per k row with 97 nodes.  1000 = one call per row: "
                                 "Sigma x6.3 faster, ImSigma changes by <= 2e-5 relative (2026-10-02)"),
    ("profile",      "turbo",    "pairbuild quadrature preset: turbo|quick|gold"),
    ("impi-eps-union", 0.0,      "PROD_UNION: half-width of the eps-lattice core copied "
                                 "onto eps = Omega in the ImPi integral (pairbuild "
                                 "--eps-union-core); 0 = off.  0.12 removes the sawtooth "
                                 "at the eps nodes (validated locally 2026-09-29)"),
    ("impi-eps-window", True,    "pairbuild --eps-window-only: evaluate the grid residual only on the eps "
                                 "nodes of the T = 0 window [0, Omega] (the thermal kernel vanishes "
                                 "outside).  Bit-identical ImPi, ~2.5x less residual work (2026-10-02); "
                                 "skipped automatically with --impi-eps-union"),
    ("lattice-dw",   1.0e-3,     "pairbuild fine spacing of the Omega/eps lattices (the "
                                 "linear block keeps its width).  2.5e-4 resolves the "
                                 "minority QP at kF at high P (FWHM 2 Z eta ~ 2e-4) and "
                                 "removes the spurious Q rows near qff (2026-09-29)"),
    ("angular-feature-nodes", -1, "pairbuild --angular-feature-nodes: angular nodes in the "
                                 "window around the partner kF (turbo 7; -1 = profile default).  "
                                 "At high P the minority pole is ~1e-4 wide and 7 nodes alias it "
                                 "(spikes in the radial integrand, 2026-10-01): 21 halves the "
                                 "spurious bump beyond qff"),
    ("p-feature-width", -1.0,    "pairbuild --p-feature-width: half-width of the dense p windows of the "
                                 "grid residual (negative = profile default, turbo 0).  0.08 with 21 "
                                 "nodes and p-feature-mode kf_only (73 nodes) resolves the well of "
                                 "A*A - A0*A0 at p = kF_up: one-step g_c error from 0.02-0.08 to <= 0.006 "
                                 "for P = 0.10-0.75 (test/energy_bubble/pwindow_probe.py, 2026-10-02); "
                                 "residual cost per Q row x3.3"),
    ("p-feature-nodes", -1,      "pairbuild --p-feature-nodes: nodes per p window (negative = profile)"),
    ("p-feature-mode", "",       "pairbuild --p-feature-mode kf_only|kf_and_shells (empty = pairbuild "
                                 "default kf_and_shells)"),
    ("sigma-omega-dense-w", 0.0, "density --sigma-omega-dense-half-width: add the w_base nodes "
                                 "with |w| <= this to the Sigma omega nodes (0 = off).  At high P "
                                 "the minority polaron band is 0.02-0.07 wide and the 97 index-"
                                 "uniform nodes are ~7e-3 apart near 0 (3 nodes across the band "
                                 "at P = 0.90)"),
    ("sigma-omega-dense-stride", 2, "take every n-th w_base node in that dense block"),
    ("sigma-k-kf-offsets", "",   "density --sigma-k-kf-offsets: d values separated by ':' (NOT commas: "
                                 "sbatch --export splits ARGS at commas), extra Sigma k "
                                 "nodes at kF*(1 -+ d) for each spin ON TOP of sigma-nk (empty = "
                                 "off).  Default nodes near kF sit at ~0.5 and ~1.6-2.2 kF at high P"),
    ("lattice-n-linear", 0,      "pairbuild --lattice-n-linear: nodi del blocco lineare dei reticoli Omega/eps (passo "
                                 "lattice-dw).  0 = quelli del profilo, riscalati per tenere la stessa larghezza.  Ad alta "
                                 "P conviene fissarli (es. 160, quanti ne ha la produzione a dw 2.5e-4) e scalare lattice-dw con 1-P: la banda QP del minoritario "
                                 "E*_F_dn ~ 0.26 (1-P) resta risolta con lo stesso numero di nodi (2026-10-06)"),
    ("lattice-n-tail", 0,         "pairbuild lattice tail nodes; 0 keeps the profile "
                                 "default (turbo 20, quick 45, gold 70)"),
    ("omega-feature-half-width", -1.0,
                                 "half-width of each pair omega feature window; negative "
                                 "keeps the legacy |mu_up-mu_dn|+0.05 choice"),
    ("omega-feature-nodes", 0,    "nodes in each pair omega feature window; 0 keeps the "
                                 "legacy profile choice (turbo 21, otherwise 81)"),
    ("omega-feature-q-min", -1.0, "lower Q used to generate omega features; negative = all"),
    ("omega-feature-q-max", -1.0, "upper Q used to generate omega features; negative = all"),
    ("omega-feature-q-max-points", 0,
                                 "maximum representative Q values used to generate omega "
                                 "features; 0 = all"),
    ("omega-mode",   "zero_and_thresholds",
                                 "pair omega features: zero|zero_and_thresholds|"
                                 "paper_thresholds.  zero (141 nodes) has no node near "
                                 "the two-body bound state and inflates the contact by "
                                 "+33% (0.581 vs 0.438 on the same seed).  NB the "
                                 "threshold feature nodes are off the 1e-3 lattice"),
    ("time-budget",  0.0,        "minutes: do not START an iteration unless the last "
                                 "one's duration still fits (0 = no limit).  Set it "
                                 "below the walltime so a job never dies mid-iteration"),
    ("q-gl-n",       16,         "SIGMA_PN_Q_GL_N, Gauss-Legendre nodes in q "
                                 "(engine default is 8)"),

    # --- physics modes -----------------------------------------------------
    ("thouless-q-mode", "qff",   "where the Thouless condition is imposed: qff, "
                                 "qff-or-zero (the higher of the two 2D candidates: qff and the "
                                 "smallest nonzero Q; the choice for high P, 2026-10-01) or "
                                 "global-max (max of ReGamma^-1(Q,0) over Q <= 2 qff, WITHDRAWN in "
                                 "2D: it chases numerical maxima).  Old note: qff "
                                 "is never the max: up to P=0.65 the max sits at "
                                 "1.03-1.1 qff, +4 delta above it at P=0.1 (dg -0.05) "
                                 "down to +0.2 at 0.65; from P~0.7 it moves to Q~0 "
                                 "(runaway at P>=0.8).  global-max is the criterion"),
    ("eta-floor",    "broad",    "eta floor mode"),
    ("high-k-sigma", "stale",    "what to do beyond k_update_max: stale|pair-contact"),
    ("ring-mode",    "exact_window", "SIGMA_PN_RING_MODE; exact_window is the "
                                 "validated fix for aliasing near the Thouless pole"),
    ("angle-exact-cut", 1,       "SIGMA_PN_ANGLE_EXACT_CUT: exact p<=pmax cut in "
                                 "theta.  The legacy masked version is a step "
                                 "function on a uniform grid, error up to 9%"),
    ("prune-keep",   0,          "keep the big files (next cubes, pairbuild tables) only for "
                                 "the last N iterations; older ones are first summarised by "
                                 "test/extract_snapshot.py + test/extract_sigma.py into "
                                 "<out>/snap/.  0 = keep everything (~34 MB per iteration)"),
    ("density-fine-pole", True,  "density --density-fine-pole: n(k) and the density with the omega integral "
                                 "redone at the QP pole on a merged geometric grid (rows within 0.3 kF), with "
                                 "the cube's own A.  Removes the spike of n(k) at kF (trapezoid on the row: "
                                 "int A up to 1.15).  Diagnostic only, does not feed back (2026-10-02)"),
    ("impi-p-graded-min", 0.0,   "pairbuild/impi_table: finestre in p del residuo graduate geometricamente verso kF "
                                 "(env IMPI_P_GRADED_MIN; distanze da questo valore a p-feature-width, "
                                 "p-feature-nodes nodi per lato).  0 = finestre uniformi.  1e-4 toglie lo spike di "
                                 "ImGamma^-1 a Omega ~ 0 (2026-10-02)"),
    ("coherent-angle-mode", "kjac", "pairbuild --coherent-angle-mode: kjac (nodi a kF NON applicati) o phipanel "
                                 "(finestra a kF, graduata se coherent-graded-min > 0)"),
    ("coherent-graded-min", 0.0, "parte coerente analitica: grappoli in k e finestra angolare graduati verso kF "
                                 "(env COHERENT_K_GRADED_MIN e COHERENT_PHI_GRADED_MIN).  1e-4 con phipanel toglie "
                                 "il dente di ReGamma^-1 dopo qff (2026-10-02)"),
    ("coherent-k-graded-n", 0,   "nodi per lato dei grappoli in k graduati della parte coerente (0 = 120)"),
    ("coherent-phi-panels", 24,  "pannelli angolari regolari della parte coerente phipanel"),
    ("control-analytic", False,  "pairbuild --control-analytic: nel residuo a griglia A*A - A0*A0 la A0 e' calcolata "
                                 "esattamente dal modello QP invece che interpolata fra le righe della cube A0.  A k grande "
                                 "le righe sono rade e il picco QP si sposta piu' della sua larghezza: l'interpolazione "
                                 "lasciava un eccesso fino a 2.5x il valore libero vicino al bordo 2 Lambda^2.  Con "
                                 "l'opzione la bolla senza taper coincide col conto statico entro 0.5 delta (P=0.90, "
                                 "2026-10-05)"),
    ("qp-e-interp-k2", False,    "env QP_E_INTERP_K2=1: nel modello QP (controllo A0 esatto e parte coerente analitica) "
                                 "l'energia fra due righe si interpola come E - k^2 lineare piu' k^2 esatto.  Con E "
                                 "lineare in k il picco fra righe rade si sposta di ~Dk^2/4: oltre k_update_max fino a "
                                 "40 larghezze (eta), e restano rasoiate isolate nella parte a griglia (2026-10-05)"),
    ("uv-guard-lambda", 0.0,     "pair_gamma --residual-qmax-lambda: righe con Q >= F Lambda (fuori dal supporto della "
                                 "bolla numerica) usano solo Gamma0.  2 chiude il circolo ultravioletto visto con qcut a "
                                 "P = 0.80 (2026-10-04).  0 = spento"),
    ("taper-qfreeze", 0.0,       "pair_gamma --residual-taper-qfreeze: per Q < F qff il taper del residuo e' quello "
                                 "della riga F qff.  0 = taper mobile (produzione).  1 = gara qff / Q ~ 0 senza la "
                                 "pendenza spuria del taper (a P = 0.85 valeva +4.5 delta a favore di Q ~ 0), riga "
                                 "qff invariata (2026-10-03)"),
    ("ref-kk-fine-step", 0.0,    "pair_gamma --ref-kk-fine-step: KK della parte di riferimento analitica su griglia "
                                 "fine (questo passo) attorno ai suoi bordi di Pauli e soglia.  0 = spento.  2.5e-4 "
                                 "toglie le oscillazioni di ReGamma^-1 a Q piccolo (gara qff / Q ~ 0 ad alta P) "
                                 "(2026-10-03)"),
    ("coherent-phi-feature-n", 21, "nodi per lato della finestra angolare a kF della parte coerente phipanel"),
    ("zero-fit-qmin", 0.02,      "thouless-q-mode qff-or-zero-fit: finestra del fit dell'altopiano a Q ~ 0 [qff]"),
    ("zero-fit-qmax", 0.15,      "thouless-q-mode qff-or-zero-fit: finestra del fit dell'altopiano a Q ~ 0 [qff]"),
    ("omega-pauli-step", 0.0,    "impi_table: blocco uniforme di nodi Omega su |Omega| <= |mu_up - mu_dn| + 0.1 con questo "
                                 "passo (env OMEGA_PAULI_STEP), dove i bordi della finestra di Pauli delle righe Q < qff "
                                 "scorrono: riduce l'errore della KK a Q piccolo (gara qff / Q ~ 0, ramo pBCS).  0 = spento"),
    ("sigma0-mode", "fermi",     "density --sigma0-mode-up/down: fermi (sigma0 = ReSigma(kF, 0): superficie di Fermi al "
                                 "kF libero, mu fisso) oppure density (spostamento costante scelto perche' n = "
                                 "(1 -+ P)/2, cioe' mu aggiustato a ogni iterazione).  density serve nel ramo pBCS ad "
                                 "alta P, dove a mu fisso n_dn va a -13% / +16% (2026-10-03)"),
    ("density-pole-subtract", False,
                                 "density --density-pole-subtract: correct n(k) and int A for the "
                                 "quadrature error of the narrow QP pole near kF (high P: density "
                                 "+0.5-0.6% -> 0, int A at kF 1.2 -> 1.00).  Reporting only (2026-10-02)"),
    ("contact-tail", False,      "replace the tail beyond k_update_max with C/k^4 "
                                 "instead of freezing the seed values"),
]

DEFAULTS = {k.replace("-", "_"): v for k, v, _ in CONFIG}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default, help_ in CONFIG:
        # argparse passa gli help attraverso il %-formatting: un % letterale
        # ("89% of the pair") diventa una conversione e fa saltare --help.
        kw = {"help": f"{help_}  [default: {default}]".replace("%", "%%")}
        if isinstance(default, bool):
            kw["action"] = "store_true" if not default else "store_false"
            if default:
                name_ = "no-" + name
                p.add_argument(f"--{name_}", dest=name.replace("-", "_"), **kw)
                continue
        elif default is None:
            kw["type"] = float
            kw["default"] = None
        else:
            kw["type"] = type(default)
            kw["default"] = default
        p.add_argument(f"--{name}", **kw)
    p.add_argument("--dump-config", action="store_true",
                   help="print the resolved configuration and exit without running")
    return p


# ---------------------------------------------------------------------------
# q grid
# ---------------------------------------------------------------------------
def smooth_multicenter_grid(a, b, n_points, centers, widths, peaks, base=1.0):
    """Nodes distributed as rho(q) = base + sum_i peak_i sech^2((q-c_i)/w_i),
    obtained by inverting the cumulative of rho: more nodes where rho is large."""
    x = np.linspace(a, b, 8192)
    rho = np.full_like(x, float(base))
    for c, w, pk in zip(centers, widths, peaks):
        rho += float(pk) / np.cosh((x - float(c)) / float(w)) ** 2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))])
    cum /= cum[-1]
    return np.interp(np.linspace(0.0, 1.0, int(n_points)), cum, x)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------
def last_complete_iteration(out):
    """Highest iterNNN that has BOTH next cubes.  A run killed halfway through an
    iteration leaves one cube behind; that iteration is discarded rather than
    resumed from, which is what makes restarting safe."""
    last, paths = 0, None
    for d in sorted(glob.glob(os.path.join(out, "iter*", "next_cubes"))):
        u = sorted(glob.glob(os.path.join(d, "*spinup*.npz")))
        n = sorted(glob.glob(os.path.join(d, "*spindown*.npz")))
        if not u or not n:
            continue
        i = int(os.path.basename(os.path.dirname(d)).replace("iter", ""))
        if i > last:
            last, paths = i, (u[0], n[0])
    return last, paths


NK4_POINTS = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def nk4_profile(cube_path, points):
    """n(k) k^4 = k^4 int_{w<=0} A(k,w) dw at a few k; the cube's w may be per-row."""
    z = np.load(cube_path, allow_pickle=True)
    k, a, w = np.asarray(z["k"]), np.asarray(z["A"]), np.asarray(z["w"])
    if w.ndim == 1:
        m = w <= 0.0
        nk = np.trapezoid(a[:, m], w[m], axis=1)
    else:
        nk = np.array([np.trapezoid(a[j][w[j] <= 0.0], w[j][w[j] <= 0.0])
                       for j in range(k.size)])
    return [(p, float(np.interp(p, k, nk * k ** 4))) for p in points]


def grab(text, key):
    for line in text.splitlines():
        if line.startswith(key):
            return float(line.split()[-1])
    return float("nan")


# ---------------------------------------------------------------------------
def prune_old_iterations(out, current, keep, log):
    """Summarise every finished iteration into <out>/snap, then delete the big
    files of those older than the last `keep`.  Resume only needs the last
    complete iteration's next_cubes, which is never touched."""
    sys.path.insert(0, os.path.join(HERE, "test"))
    import extract_snapshot
    import extract_sigma
    extract_snapshot.extract(out)
    extract_sigma.extract(out)
    freed = 0
    for itd in sorted(glob.glob(os.path.join(out, "iter[0-9][0-9][0-9]"))):
        j = int(os.path.basename(itd)[4:])
        if j > current - keep:
            continue
        tag = os.path.basename(itd)[4:]
        snap = os.path.join(out, "snap")
        if not (os.path.exists(os.path.join(snap, f"iter{tag}.npz"))
                and os.path.exists(os.path.join(snap, f"sigma{tag}.npz"))):
            continue            # never delete what has not been summarised
        for pat in ("next_cubes/*.npz", "pairbuild/impi/*.npz", "pairbuild/pair/*.npz",
                    "pairbuild/control/*.npz", "pairbuild/impi/residual/*.npz",
                    "density/*cubes*/*.npz"):
            for f in glob.glob(os.path.join(itd, pat)):
                freed += os.path.getsize(f)
                os.remove(f)
    if freed:
        log(f"=== iter {current} PRUNE: freed {freed / 2**20:.0f} MB (kept last {keep}) ===")


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    out = os.path.abspath(cfg["out"])

    start, resumed = 1, None
    if cfg["resume"]:
        last, resumed = last_complete_iteration(out)
        start = last + 1

    if resumed:
        up, down = resumed
        origin = f"resumed from iteration {start - 1}"
    elif cfg["up"] and cfg["down"]:
        up, down = os.path.abspath(cfg["up"]), os.path.abspath(cfg["down"])
        origin = "explicit --up/--down"
    else:
        sd = os.path.join(HERE, cfg["seed_dir"])
        u = sorted(glob.glob(os.path.join(sd, "*spinup*.npz")))
        n = sorted(glob.glob(os.path.join(sd, "*spindown*.npz")))
        if not u or not n:
            raise SystemExit(f"no seed cubes in {sd}; pass --up/--down")
        up, down, origin = u[0], n[0], f"warm seed from {cfg['seed_dir']}/"

    if cfg["pair_shift_freeze"] and cfg["pair_shift_fixed"] is not None:
        raise SystemExit("--pair-shift-fixed and --pair-shift-freeze are exclusive")
    n_iter = cfg["iters"]
    if cfg["target"]:
        n_iter = min(n_iter, max(0, cfg["target"] - (start - 1)))
        if n_iter == 0:
            print(f"already at {start - 1}/{cfg['target']} iterations: nothing to do")
            return 0

    # --- physics read off the cube, not configured ------------------------
    zu = np.load(up)
    zd = np.load(down)
    mu_up = float(np.asarray(zu["mu_up"]).reshape(-1)[0])
    mu_dn = float(np.asarray(zd["mu_dn" if "mu_dn" in zd.files else "mu_down"]
                            ).reshape(-1)[0])
    qff = float(np.sqrt(mu_up) - np.sqrt(mu_dn))
    cube_k_max = float(np.asarray(zu["k"]).reshape(-1)[-1])

    # the short blanket: the bubble needs |Q - k_up| on the cube for |k_up| < Lambda
    q_table_max = min(cfg["q_cap"], cube_k_max - cfg["bubble_lambda"] - 0.1)
    k_update_max = (cfg["k_update_max"] if cfg["k_update_max"] > 0
                    else q_table_max - cfg["pintmax"])
    if k_update_max > cfg["pintmax"] - 3.0:
        print(f"[warn] k_update_max={k_update_max:g} > pintmax-3={cfg['pintmax'] - 3:g}: "
              f"Sigma's tail channel is cut above k~pintmax+1.5 inside the update window")

    tail_n = max(cfg["tail_n"],
                 int(round(cfg["tail_n"] * (q_table_max - 2.2) / (11.5 - 2.2))))

    resolved = dict(
        cfg, _start_iter=start, _n_iter=n_iter, _origin=origin,
        _up=up, _down=down, _mu_up=mu_up, _mu_dn=mu_dn, _qff=qff,
        _cube_k_max=cube_k_max, _q_table_max=q_table_max,
        _k_update_max=k_update_max, _tail_n_rescaled=tail_n,
        _sigma_evals_per_spin=cfg["sigma_nk"] * cfg["sigma_nomega"],
    )

    print(json.dumps(resolved, indent=2, default=str))
    if args.dump_config:
        return 0

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(resolved, fh, indent=2, default=str)

    env = dict(os.environ)
    env["SIGMA_PN_PINTMAX"] = f"{cfg['pintmax']:g}"
    env["SIGMA_PN_RING_MODE"] = cfg["ring_mode"]
    env["SIGMA_PN_ANGLE_EXACT_CUT"] = str(cfg["angle_exact_cut"])
    env["SIGMA_PN_Q_GL_N"] = str(cfg["q_gl_n"])
    if cfg["omega_pauli_step"] > 0:
        env["OMEGA_PAULI_STEP"] = f"{cfg['omega_pauli_step']:g}"
    if cfg["impi_p_graded_min"] > 0:
        env["IMPI_P_GRADED_MIN"] = f"{cfg['impi_p_graded_min']:g}"
    if cfg["coherent_graded_min"] > 0:
        env["COHERENT_K_GRADED_MIN"] = f"{cfg['coherent_graded_min']:g}"
        env["COHERENT_PHI_GRADED_MIN"] = f"{cfg['coherent_graded_min']:g}"
    if cfg["qp_e_interp_k2"]:
        env["QP_E_INTERP_K2"] = "1"
    if cfg["coherent_k_graded_n"] > 0:
        env["COHERENT_K_GRADED_N"] = str(cfg["coherent_k_graded_n"])
    env["COHERENT_PHI_PANELS"] = str(cfg["coherent_phi_panels"])
    env["COHERENT_PHI_FEATURE_N"] = str(cfg["coherent_phi_feature_n"])
    if cfg["pair_shift_fixed"] is not None:
        env["PAIR_SHIFT_FIXED"] = f"{cfg['pair_shift_fixed']:.15g}"
    # the loop is process-parallel; leave BLAS single-threaded or the pool and the
    # threads fight over the same cores
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env.setdefault(v, "1")

    log_path = os.path.join(out, "loop.log")
    logf = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== {origin}; iterations {start}..{start + n_iter - 1} ===")
    log(f"=== mu_up={mu_up:.6f} mu_dn={mu_dn:.6f} qff={qff:.10f} "
        f"cube_k_max={cube_k_max:.4f} -> q_table_max={q_table_max:.4f} "
        f"k_update_max={k_update_max:.4f} ===")

    frozen_path = os.path.join(out, "frozen_shift.txt")

    def frozen_shift():
        try:
            return float(open(frozen_path).read().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    contacts = []
    last_minutes = 0.0
    t_job = time.time()
    for i in range(start, start + n_iter):
        if cfg["time_budget"] > 0:
            used = (time.time() - t_job) / 60.0
            if used + 1.15 * last_minutes > cfg["time_budget"]:
                log(f"=== time budget: {used:.1f} min used, last iteration took "
                    f"{last_minutes:.1f}; stopping before iter {i} ===")
                break
        if cfg["pair_shift_freeze"]:
            fs = frozen_shift()
            if fs is not None:
                env["PAIR_SHIFT_FIXED"] = f"{fs:.15g}"
            else:
                env.pop("PAIR_SHIFT_FIXED", None)
        t0 = time.time()
        itd = os.path.join(out, f"iter{i:03d}")
        os.makedirs(itd, exist_ok=True)

        q_core = smooth_multicenter_grid(0.0, 2.2, cfg["core_n"],
                                         [0.0, qff], [0.15, 0.06], [12.0, 15.0])
        q_tail = smooth_multicenter_grid(2.2, q_table_max, tail_n,
                                         [3.0], [0.6], [6.0])
        q_grid = np.unique(np.concatenate([q_core, q_tail]))

        pb = os.path.join(itd, "pairbuild")
        feature_half_width = (
            cfg["omega_feature_half_width"]
            if cfg["omega_feature_half_width"] >= 0.0
            else abs(mu_up - mu_dn) + 0.05
        )
        feature_nodes = (
            cfg["omega_feature_nodes"]
            if cfg["omega_feature_nodes"] > 0
            else (21 if cfg["profile"] == "turbo" else 81)
        )
        log(f"=== iter {i} START PAIRBUILD: profile={cfg['profile']} "
            f"Lambda={cfg['bubble_lambda']:g} p_nodes={cfg['p_nodes'] or 'profile'} "
            f"omega_feature={cfg['omega_mode']} width={feature_half_width:g} "
            f"nodes={feature_nodes} max_q_points={cfg['omega_feature_q_max_points']}"
            + (f" PROD_UNION eps_union={cfg['impi_eps_union']:g}" if cfg["impi_eps_union"] > 0 else "")
            + (f" lattice_dw={cfg['lattice_dw']:g}" if abs(cfg["lattice_dw"] - 1.0e-3) > 1e-12 else "")
            + (f" ang_feat={cfg['angular_feature_nodes']}" if cfg["angular_feature_nodes"] >= 0 else "")
            + (f" p_feat={cfg['p_feature_width']:g}/{cfg['p_feature_nodes']}/{cfg['p_feature_mode'] or 'kf_and_shells'}"
               if cfg["p_feature_width"] >= 0 else "")
            + (f" sigma_dense_w={cfg['sigma_omega_dense_w']:g}/{cfg['sigma_omega_dense_stride']}"
               if cfg["sigma_omega_dense_w"] > 0 else "")
            + (f" sigma_k_kf={cfg['sigma_k_kf_offsets']}" if cfg["sigma_k_kf_offsets"] else "")
            + " ===")
        cmd = [sys.executable, "-m", "fflo.pairbuild",
               "--up", up, "--down", down, "--out-dir", pb,
               "--workers", str(cfg["workers"]), "--profile", cfg["profile"],
               "--q-points", ",".join(f"{v:.15g}" for v in q_grid),
               "--q-table-max", f"{q_table_max:.15g}",
               "--omega-feature-mode", cfg["omega_mode"],
               "--omega-feature-half-width", f"{feature_half_width:.15g}",
               "--omega-feature-nodes", str(feature_nodes)]
        if cfg["lattice_n_tail"] > 0:
            cmd += ["--lattice-n-tail", str(cfg["lattice_n_tail"])]
        if cfg["lattice_n_linear"] > 0:
            cmd += ["--lattice-n-linear", str(cfg["lattice_n_linear"])]
        if cfg["angular_feature_nodes"] >= 0:
            cmd += ["--angular-feature-nodes", str(cfg["angular_feature_nodes"])]
        if cfg["p_feature_width"] >= 0:
            cmd += ["--p-feature-width", f"{cfg['p_feature_width']:.15g}"]
        if cfg["p_feature_nodes"] >= 0:
            cmd += ["--p-feature-nodes", str(cfg["p_feature_nodes"])]
        if cfg["p_feature_mode"]:
            cmd += ["--p-feature-mode", cfg["p_feature_mode"]]
        if cfg["impi_eps_union"] > 0:
            cmd += ["--eps-union-core", f"{cfg['impi_eps_union']:.15g}"]
        elif cfg["impi_eps_window"]:
            cmd += ["--eps-window-only"]
        if abs(cfg["lattice_dw"] - 1.0e-3) > 1e-12:
            cmd += ["--lattice-dw", f"{cfg['lattice_dw']:.15g}"]
        if cfg["omega_feature_q_min"] >= 0.0:
            cmd += ["--omega-feature-q-min", f"{cfg['omega_feature_q_min']:.15g}"]
        if cfg["omega_feature_q_max"] >= 0.0:
            cmd += ["--omega-feature-q-max", f"{cfg['omega_feature_q_max']:.15g}"]
        if cfg["omega_feature_q_max_points"] > 0:
            cmd += ["--omega-feature-q-max-points",
                    str(cfg["omega_feature_q_max_points"])]
        if cfg["bubble_lambda"] != 4.0:
            # only when needed: a pairbuild without this option keeps working at 4
            cmd += ["--bubble-p-int-max", f"{cfg['bubble_lambda']:.15g}"]
        if cfg["p_nodes"] > 0:
            cmd += ["--p-nodes", str(cfg["p_nodes"])]
        if cfg["coherent_nk"] > 0:
            cmd += ["--coherent-nk", str(cfg["coherent_nk"])]
        if cfg["coherent_angle_mode"] != "kjac":
            cmd += ["--coherent-angle-mode", cfg["coherent_angle_mode"]]
        if cfg["ref_kk_fine_step"] > 0.0:
            cmd += ["--ref-kk-fine-step", f"{cfg['ref_kk_fine_step']:.15g}"]
        if cfg["taper_qfreeze"] > 0.0:
            cmd += ["--residual-taper-qfreeze", f"{cfg['taper_qfreeze']:.15g}"]
        if cfg["uv_guard_lambda"] > 0.0:
            cmd += ["--residual-qmax-lambda", f"{cfg['uv_guard_lambda']:.15g}"]
        if cfg["control_analytic"]:
            cmd += ["--control-analytic"]
        fix = cfg["truncation_fix"]
        split = (fix != "none" or cfg["taper_mode"] != "qkin_cosine"
                 or cfg["taper_stop"] != 20.0 or cfg["im_sign_guard"] != "on")
        if split:
            # pairbuild stops at impi/impi_table.npz; the fix and pair_gamma follow
            cmd.append("--skip-pair")
        with open(os.path.join(itd, "pairbuild.log"), "w") as fh:
            rc = subprocess.run(cmd, env=env, cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"=== iter {i}: PAIRBUILD FAILED rc={rc} ===")
            return 1
        if split:
            impi = os.path.join(pb, "impi", "impi_table.npz")
            steps = []
            if fix != "none":
                fixed = os.path.join(pb, "impi", "impi_table_truncfix.npz")
                steps.append([sys.executable, os.path.join(HERE, "test", "fix_truncation.py"),
                              "--mu-mode", fix, impi, fixed])
            else:
                fixed = impi
            steps.append(
                # the pair_gamma call inside fflo/pairbuild.py, with the test knobs
                [sys.executable, "-m", "fflo.pair_gamma",
                 "--no-plots", "--impi-table-path", fixed,
                 "--kk-pad", "0", "--kk-oversample", "1", "--kk-tail", "zero",
                 "--kk-tail-alpha", "nan", "--kk-max-n", "0",
                 "--eta-gamma", "0.002", "--im-inv-sign-guard", cfg["im_sign_guard"],
                 "--reference-mode", "proxy_residual",
                 "--qp-residual-mode", "cross_inc", "--qpqp-source", "auto",
                 "--qpqp-xi-cutoff", "0.015", "--qpqp-n-p", "5000",
                 "--qpqp-n-phi", "24", "--residual-omega-taper-start", "0",
                 "--residual-omega-taper-stop", f"{cfg['taper_stop']:g}",
                 "--residual-omega-taper-mode", cfg["taper_mode"],
                 "--out-dir", os.path.join(pb, "pair")]
                + (["--ref-kk-fine-step", f"{cfg['ref_kk_fine_step']:.15g}"]
                   if cfg["ref_kk_fine_step"] > 0.0 else [])
                + (["--residual-taper-qfreeze", f"{cfg['taper_qfreeze']:.15g}"]
                   if cfg["taper_qfreeze"] > 0.0 else [])
                + (["--residual-qmax-lambda", f"{cfg['uv_guard_lambda']:.15g}"]
                   if cfg["uv_guard_lambda"] > 0.0 else []))
            with open(os.path.join(itd, "truncfix_pair.log"), "w") as fh:
                for step in steps:
                    rc = subprocess.run(step, env=env, cwd=HERE, stdout=fh,
                                        stderr=subprocess.STDOUT).returncode
                    if rc:
                        log(f"=== iter {i}: TRUNCATION FIX / PAIR_GAMMA FAILED rc={rc} ===")
                        return 1
            if fix != "none":
                with open(os.path.join(itd, "truncfix_pair.log")) as fh:
                    note = [l.strip() for l in fh if l.startswith("Lambda=")]
                log(f"=== iter {i} TRUNCFIX: {note[0] if note else '?'} ===")
            log(f"=== iter {i} PAIR_GAMMA: taper {cfg['taper_mode']} stop "
                f"{cfg['taper_stop']:g}, im-sign-guard {cfg['im_sign_guard']} ===")

        pair_table = os.path.join(pb, "pair", "pair_gamma_table.npz")
        t = np.load(pair_table)
        q_t = np.asarray(t["q"]).reshape(-1)
        w_t = np.asarray(t["omega"]).reshape(-1)
        iw = int(np.argmin(np.abs(w_t)))
        log(f"=== iter {i} ReInvGamma(w~0): Q=0 -> "
            f"{float(t['ReInvGamma'][int(np.argmin(np.abs(q_t))), iw]):.6f} | "
            f"Q=qff -> "
            f"{float(t['ReInvGamma'][int(np.argmin(np.abs(q_t - qff))), iw]):.6f} | "
            f"nQ={q_t.size} ===")

        dd, nc = os.path.join(itd, "density"), os.path.join(itd, "next_cubes")
        cmd = [sys.executable, "-m", "fflo.density",
               "--pair-table", pair_table, "--seed-up", up, "--seed-down", down,
               "--subcritical-delta", f"{cfg['delta']:.15g}",
               "--sigma-nk", str(cfg["sigma_nk"]),
               "--n-theta", str(cfg["n_theta"]),
               "--omega-chunk", str(cfg["omega_chunk"]),
               "--sigma-nomega", str(cfg["sigma_nomega"]),
               "--eta-floor-mode", cfg["eta_floor"],
               "--thouless-q-mode", cfg["thouless_q_mode"],
               "--zero-fit-qmin", f"{cfg['zero_fit_qmin']:g}", "--zero-fit-qmax", f"{cfg['zero_fit_qmax']:g}",
               "--high-k-sigma-mode", cfg["high_k_sigma"],
               "--sigma-k-feature-fraction", "0", "--pole-refind-n-local", "61",
               "--k-update-max", f"{k_update_max:.15g}"]
        if k_update_max > q_table_max - cfg["pintmax"] + 1e-12:
            # density would otherwise clip the override back to q_table_max - pintmax
            cmd.append("--allow-partial-q-support")
        if cfg["contact_tail"]:
            cmd.append("--density-contact-tail")
        if cfg["sigma0_mode"] != "fermi":
            cmd += ["--sigma0-mode-up", cfg["sigma0_mode"], "--sigma0-mode-down", cfg["sigma0_mode"]]
        if cfg["density_fine_pole"]:
            cmd.append("--density-fine-pole")
        elif cfg["density_pole_subtract"]:
            cmd.append("--density-pole-subtract")
        if cfg["sigma_k_kf_offsets"]:
            cmd += ["--sigma-k-kf-offsets", cfg["sigma_k_kf_offsets"]]
        if cfg["sigma_omega_dense_w"] > 0:
            cmd += ["--sigma-omega-dense-half-width", f"{cfg['sigma_omega_dense_w']:.15g}",
                    "--sigma-omega-dense-stride", str(cfg["sigma_omega_dense_stride"])]
        cmd += ["--mix", f"{cfg['alpha']:g}", "1.0", "--out-dir", dd,
                "--emit-cube-dir", nc, "--emit-cube-alpha", f"{cfg['alpha']:g}",
                "--iteration-index", str(i), "--workers", str(cfg["workers"])]
        with open(os.path.join(itd, "density.log"), "w") as fh:
            rc = subprocess.run(cmd, env=env, cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"=== iter {i}: DENSITY FAILED rc={rc} ===")
            return 1

        txt = open(os.path.join(dd, "density_summary.txt")).read()
        # the stability diagnostic is the PAIR channel, not the gap: in a run that
        # converges pair_contact contracts 1-2% per iteration; if it grows, the
        # Sigma -> Pi -> Gamma -> Sigma feedback is diverging
        pc = grab(txt, "pair_contact ")
        ratio = pc / contacts[-1] if contacts else float("nan")
        contacts.append(pc)
        raw_shift = grab(txt, "pair_raw_thouless_shift ")
        q_sel = grab(txt, "pair_q_selected ")
        log(f"=== iter {i} PAIR: contact={pc:.6f} (x{ratio:.4f} vs previous) "
            f"shift={raw_shift:+.6f} Q={q_sel:.4f}"
            f"{' FROZEN=' + env['PAIR_SHIFT_FIXED'] if 'PAIR_SHIFT_FIXED' in env else ''} ===")
        if cfg["pair_shift_freeze"] and frozen_shift() is None and np.isfinite(raw_shift):
            with open(frozen_path, "w") as fh:
                fh.write(f"{raw_shift:.15g}  # raw Thouless shift of iter {i}, frozen from here on\n")
            log(f"=== shift frozen at {raw_shift:+.15g} from iter {i} ===")
        for line in txt.splitlines():
            if line.startswith(f"alpha_{cfg['alpha']:g} "):
                p = line.split()
                log(f"=== iter {i} GAP: up={p[3]} down={p[6]} ===")
        last_minutes = (time.time() - t0) / 60.0
        log(f"=== iter {i} TIME: {last_minutes:.1f} min ===")

        nu = sorted(glob.glob(os.path.join(nc, "*spinup*.npz")))
        nd = sorted(glob.glob(os.path.join(nc, "*spindown*.npz")))
        if not nu or not nd:
            log(f"=== iter {i}: next cubes missing ===")
            return 1
        # the tail, where the pumping shows: n(k) k^4 of the minority spin
        try:
            log(f"=== iter {i} NK4dn: " + " ".join(
                f"k{kk:g}={v:.3f}" for kk, v in nk4_profile(nd[0], NK4_POINTS)) + " ===")
        except Exception as exc:  # diagnostics must never kill the loop
            log(f"=== iter {i} NK4dn: unavailable ({exc}) ===")
        up, down = nu[0], nd[0]
        if cfg["prune_keep"] > 0:
            try:
                prune_old_iterations(out, i, int(cfg["prune_keep"]), log)
            except Exception as exc:  # disk hygiene must never kill the loop
                log(f"=== iter {i} PRUNE: skipped ({exc}) ===")

    log("=== LOOP DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
