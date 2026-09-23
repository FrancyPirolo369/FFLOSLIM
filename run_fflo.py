#!/usr/bin/env python3
"""The FFLO self-consistency loop: one entry point, one place for every default.

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

HERE = os.path.dirname(os.path.abspath(__file__))

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

    # --- the budget --------------------------------------------------------
    ("pintmax",      6.0,        "SIGMA_PN_PINTMAX: internal momentum cutoff in "
                                 "Sigma.  NOT an accuracy knob -- 89% of the pair "
                                 "weight sits at Q<=2, so the dominant internal "
                                 "momentum is p~k and this cancels the dominant "
                                 "channel for k > pintmax+1"),
    ("q-cap",        20.0,       "ceiling on q_table_max; the real limit is "
                                 "k_max_cube - 4.1 (pairbuild's --p-int-max is 4)"),

    # --- resolution --------------------------------------------------------
    ("core-n",       40,         "q nodes in [0, 2.2], packed around 0 and qff"),
    ("tail-n",       19,         "q nodes above 2.2 (auto-rescaled with the span)"),
    ("sigma-nk",     24,         "k points where Sigma is really evaluated"),
    ("sigma-nomega", 41,         "omega points where Sigma is really evaluated; "
                                 "cost is sigma_nk * sigma_nomega per spin"),
    ("n-theta",      32,         "angular nodes in the density stage"),
    ("profile",      "turbo",    "pairbuild quadrature preset: turbo|quick|gold"),
    ("q-gl-n",       16,         "SIGMA_PN_Q_GL_N, Gauss-Legendre nodes in q "
                                 "(engine default is 8)"),

    # --- physics modes -----------------------------------------------------
    ("thouless-q-mode", "qff",   "where the Thouless condition is imposed.  Use qff; "
                                 "global-max pins Q~0.15 and collapses n_down by 80%"),
    ("eta-floor",    "broad",    "eta floor mode"),
    ("high-k-sigma", "stale",    "what to do beyond k_update_max: stale|pair-contact"),
    ("ring-mode",    "exact_window", "SIGMA_PN_RING_MODE; exact_window is the "
                                 "validated fix for aliasing near the Thouless pole"),
    ("angle-exact-cut", 1,       "SIGMA_PN_ANGLE_EXACT_CUT: exact p<=pmax cut in "
                                 "theta.  The legacy masked version is a step "
                                 "function on a uniform grid, error up to 9%"),
    ("contact-tail", False,      "replace the tail beyond k_update_max with C/k^4 "
                                 "instead of freezing the seed values"),
]

DEFAULTS = {k.replace("-", "_"): v for k, v, _ in CONFIG}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default, help_ in CONFIG:
        kw = {"help": f"{help_}  [default: {default}]"}
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


def grab(text, key):
    for line in text.splitlines():
        if line.startswith(key):
            return float(line.split()[-1])
    return float("nan")


# ---------------------------------------------------------------------------
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

    # the short blanket: pairbuild hardcodes --p-int-max 4
    q_table_max = min(cfg["q_cap"], cube_k_max - 4.0 - 0.1)
    k_update_max = q_table_max - cfg["pintmax"]

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

    contacts = []
    for i in range(start, start + n_iter):
        t0 = time.time()
        itd = os.path.join(out, f"iter{i:03d}")
        os.makedirs(itd, exist_ok=True)

        q_core = smooth_multicenter_grid(0.0, 2.2, cfg["core_n"],
                                         [0.0, qff], [0.15, 0.06], [12.0, 15.0])
        q_tail = smooth_multicenter_grid(2.2, q_table_max, tail_n,
                                         [3.0], [0.6], [6.0])
        q_grid = np.unique(np.concatenate([q_core, q_tail]))

        pb = os.path.join(itd, "pairbuild")
        cmd = [sys.executable, "-m", "fflo.pairbuild",
               "--up", up, "--down", down, "--out-dir", pb,
               "--workers", str(cfg["workers"]), "--profile", cfg["profile"],
               "--q-points", ",".join(f"{v:.15g}" for v in q_grid),
               "--q-table-max", f"{q_table_max:.15g}",
               "--omega-feature-mode", "zero",
               "--omega-feature-half-width", f"{abs(mu_up - mu_dn) + 0.05:.6f}",
               "--omega-feature-nodes", "21" if cfg["profile"] == "turbo" else "81"]
        with open(os.path.join(itd, "pairbuild.log"), "w") as fh:
            rc = subprocess.run(cmd, env=env, cwd=HERE, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc:
            log(f"=== iter {i}: PAIRBUILD FAILED rc={rc} ===")
            return 1

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
               "--sigma-nomega", str(cfg["sigma_nomega"]),
               "--eta-floor-mode", cfg["eta_floor"],
               "--thouless-q-mode", cfg["thouless_q_mode"],
               "--high-k-sigma-mode", cfg["high_k_sigma"],
               "--sigma-k-feature-fraction", "0", "--pole-refind-n-local", "61"]
        if cfg["contact_tail"]:
            cmd.append("--density-contact-tail")
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
        log(f"=== iter {i} PAIR: contact={pc:.6f} (x{ratio:.4f} vs previous) "
            f"shift={grab(txt, 'pair_raw_thouless_shift '):+.6f} ===")
        for line in txt.splitlines():
            if line.startswith(f"alpha_{cfg['alpha']:g} "):
                p = line.split()
                log(f"=== iter {i} GAP: up={p[3]} down={p[6]} ===")
        log(f"=== iter {i} TIME: {(time.time() - t0) / 60:.1f} min ===")

        nu = sorted(glob.glob(os.path.join(nc, "*spinup*.npz")))
        nd = sorted(glob.glob(os.path.join(nc, "*spindown*.npz")))
        if not nu or not nd:
            log(f"=== iter {i}: next cubes missing ===")
            return 1
        up, down = nu[0], nd[0]

    log("=== LOOP DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
