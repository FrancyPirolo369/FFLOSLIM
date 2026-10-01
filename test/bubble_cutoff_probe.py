#!/usr/bin/env python3
"""Cheap one-shot estimate of pair-bubble cutoff saturation.

This is deliberately a diagnostic, not a production replacement.  It runs the
matched product on a handful of Q values and a sparse omega grid once, keeps
the radial p integrand, and reads all requested cutoffs from its cumulative.
The analytic QP add-back and the proxy residual reconstruction are then
repeated cheaply for every cutoff.  The reported cutoff error is therefore
the estimated change of Re Gamma^{-1}(Q, 0), not Im Pi(Q, 0) (which is often
zero and is a useless cutoff diagnostic).

Typical use, from X_FFLO_SLIM::

    ../.venv/bin/python test/bubble_cutoff_probe.py \
      --up seeds/A_komega_spinup_warm.npz \
      --down seeds/A_komega_spindown_warm.npz \
      --out-dir out/cutoff_probe

The default probe uses Q={0,Q_FFLO,Q_FFLO+0.1}, Lambda={2,3,4,5,6,8,10,12},
the production internal-Sigma rebuild when available, and the production
qkin_cosine residual taper.  It writes cutoff_probe.tsv and cutoff_probe.npz.
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fflo.adn_phi import (  # noqa: E402
    build_eps_grid,
    build_phipanel_geometry_cache,
    enrich_grid_with_feature_windows,
    load_akw_cube,
    precompute_kjac_integrated_slice,
)
from fflo.impi_cv import _canonical_qp_fields, _cv_init, _cv_work, _kf  # noqa: E402
from fflo.impi_table import _load_sigma_rebuild_fields, _rebuild_a_bilinear  # noqa: E402
from fflo.kramers_kronig import (  # noqa: E402
    _thermal_kernel_scalar_zero_temp,
    _trapz_axis_panel_kernel,
    bilinear_eval_with_support,
    g0_inverse,
    load_cube_a_convention,
    load_cube_physics,
    normalize_spectral_rows,
    proxy_full_pi_from_gamma,
    proxy_medium_pi_from_gamma,
)
from fflo.pair_gamma import kk_on_tapered_residual_native  # noqa: E402
from fflo.qp_cube import build_qp_cube, qp_model_from_cube  # noqa: E402


_GRID_CTX: dict[str, object] = {}


def _csv_floats(raw: str) -> np.ndarray:
    values = [float(x.strip()) for x in str(raw).replace(":", ",").split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated number")
    out = np.asarray(sorted(set(values)), dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("all list values must be finite")
    return out


def _npz_has(path: Path, *names: str) -> bool:
    with np.load(path) as data:
        return all(name in data.files for name in names)


def _common_w_bounds(w_axis: np.ndarray) -> tuple[float, float]:
    w = np.asarray(w_axis, dtype=float)
    if w.ndim == 1:
        return float(w[0]), float(w[-1])
    return float(np.max(w[:, 0])), float(np.min(w[:, -1]))


def _omega_grid(
    q_values: np.ndarray,
    mu_up: float,
    mu_dn: float,
    *,
    core: float,
    span: float,
    n_core: int,
    n_tail: int,
) -> np.ndarray:
    core_f = float(core)
    span_f = float(span)
    if not (0.0 < core_f <= span_f):
        raise ValueError("need 0 < omega-core <= omega-span")
    n_core_i = max(9, int(n_core))
    if n_core_i % 2 == 0:
        n_core_i += 1
    grid = np.linspace(-core_f, core_f, n_core_i, dtype=float)
    if span_f > core_f and int(n_tail) > 0:
        tail = np.geomspace(core_f, span_f, int(n_tail) + 1, dtype=float)[1:]
        grid = np.concatenate((-tail[::-1], grid, tail))

    # Pair thresholds and q-kinematic taper onset are the only omega features
    # needed for this deliberately coarse comparison.
    kf_up = float(np.sqrt(max(mu_up, 0.0)))
    features: list[float] = [0.0]
    for qv in np.asarray(q_values, dtype=float):
        features.append(0.5 * qv * qv - mu_up - mu_dn)
        features.append((qv + kf_up) ** 2 - mu_dn)
    return enrich_grid_with_feature_windows(
        np.unique(grid),
        a=-span_f,
        b=span_f,
        features=features,
        half_width=min(0.25, 0.2 * core_f),
        n_local=7,
    )


def _p_grid(
    cutoffs: np.ndarray,
    q_values: np.ndarray,
    mu_up: float,
    mu_dn: float,
    *,
    p_min: float,
    core_max: float,
    n_core: int,
    n_tail: int,
) -> np.ndarray:
    p_max = float(cutoffs[-1])
    p_core_max = min(float(core_max), p_max)
    core = np.linspace(float(p_min), p_core_max, max(9, int(n_core)), dtype=float)
    pieces = [core, np.asarray(cutoffs, dtype=float)]
    if p_max > p_core_max and int(n_tail) >= 2:
        tail_lo = max(p_core_max, max(float(p_min), 1.0e-8))
        pieces.append(np.geomspace(tail_lo, p_max, int(n_tail), dtype=float))
    grid = np.unique(np.concatenate(pieces))

    kf_up = float(np.sqrt(max(mu_up, 0.0)))
    kf_dn = float(np.sqrt(max(mu_dn, 0.0)))
    features: list[float] = [kf_up, kf_dn]
    for qv in np.asarray(q_values, dtype=float):
        features.extend((abs(qv - kf_up), abs(qv - kf_dn), qv + kf_up, qv + kf_dn))
    return enrich_grid_with_feature_windows(
        grid,
        a=float(p_min),
        b=p_max,
        features=features,
        half_width=0.05,
        n_local=7,
    )


def _cumulative_trapezoid_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    vals = np.asarray(y, dtype=float)
    axis = np.asarray(x, dtype=float)
    out = np.zeros_like(vals)
    out[..., 1:] = np.cumsum(
        0.5 * (vals[..., 1:] + vals[..., :-1]) * np.diff(axis),
        axis=-1,
    )
    return out


def _grid_q_worker(task: tuple[int, float]) -> tuple[int, np.ndarray, float]:
    """Return raw cumulative matched-residual integrals for one Q row."""
    iq, q_value = task
    c = _GRID_CTX
    p = np.asarray(c["p"], dtype=float)
    omega = np.asarray(c["omega"], dtype=float)
    eps = np.asarray(c["eps"], dtype=float)
    cutoffs = np.asarray(c["cutoffs"], dtype=float)
    kd = np.asarray(c["kd"], dtype=float)
    wd = np.asarray(c["wd"], dtype=float)
    ad = np.asarray(c["ad"], dtype=float)
    control_kd = np.asarray(c["control_kd"], dtype=float)
    control_wd = np.asarray(c["control_wd"], dtype=float)
    control_ad = np.asarray(c["control_ad"], dtype=float)
    aup = np.asarray(c["aup"], dtype=float)
    control_up = np.asarray(c["control_up"], dtype=float)

    k_features = [float(x) for x in c["k_features"]]
    geom = build_phipanel_geometry_cache(
        kd,
        p,
        q_value=float(q_value),
        routing="difference",
        n_phi_panel=int(c["angular_nodes"]),
        phi_panel_chunk_size=int(c["angular_chunk"]),
        k_features=k_features,
        k_feature_n_local=int(c["angular_feature_nodes"]),
        k_feature_half_width=0.05,
    )
    residual_raw = np.zeros((cutoffs.size, omega.size), dtype=float)
    support_sum = 0.0
    re_dn_obj = c.get("re_dn")
    im_dn_obj = c.get("im_dn")
    re_dn = None if re_dn_obj is None else np.asarray(re_dn_obj, dtype=float)
    im_dn = None if im_dn_obj is None else np.asarray(im_dn_obj, dtype=float)

    for io, omega_value in enumerate(omega):
        full_down, support = precompute_kjac_integrated_slice(
            kd,
            wd,
            ad,
            p,
            eps,
            q_value=float(q_value),
            omega_value=float(omega_value),
            routing="difference",
            n_kquad=int(c["angular_nodes"]),
            kquad_chunk_size=int(c["angular_chunk"]),
            geom_cache=geom,
            re_sigma=re_dn,
            im_sigma=im_dn,
            sigma0=float(c["sigma0_dn"]),
            mu=float(c["mu_dn"]),
            eta=float(c["eta_dn"]),
        )
        control_down, _ = precompute_kjac_integrated_slice(
            control_kd,
            control_wd,
            control_ad,
            p,
            eps,
            q_value=float(q_value),
            omega_value=float(omega_value),
            routing="difference",
            n_kquad=int(c["angular_nodes"]),
            kquad_chunk_size=int(c["angular_chunk"]),
            geom_cache=geom,
        )
        product_residual = aup * full_down - control_up * control_down
        i_p = _trapz_axis_panel_kernel(
            product_residual,
            eps,
            float(omega_value),
            _thermal_kernel_scalar_zero_temp,
        )
        cumulative = _cumulative_trapezoid_rows(p * i_p, p)
        residual_raw[:, io] = np.interp(cutoffs, p, cumulative)
        support_sum += float(np.mean(support))

    print(f"[grid] Q={q_value:.9g}: {omega.size} omega slices complete", flush=True)
    return int(iq), residual_raw, support_sum / max(1, omega.size)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot quick-and-dirty saturation probe for the pair-bubble p cutoff."
    )
    p.add_argument("--up", type=Path, required=True, help="Spin-up input cube.")
    p.add_argument("--down", type=Path, required=True, help="Spin-down input cube.")
    p.add_argument("--control-up", type=Path, default=None, help="Existing matched A0_up cube.")
    p.add_argument("--control-down", type=Path, default=None, help="Existing matched A0_down cube.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "out/cutoff_probe")
    p.add_argument("--cutoffs", default="2,3,4,5,6,8,10,12")
    p.add_argument(
        "--q",
        default="auto",
        help="Comma-separated Q probes, or 'auto' for 0,QFFLO,QFFLO+q-offset.",
    )
    p.add_argument("--q-offset", type=float, default=0.1)
    p.add_argument("--line-mode", choices=("auto", "rebuild", "stored"), default="auto")
    p.add_argument("--workers", type=int, default=3)

    p.add_argument("--omega-core", type=float, default=24.0)
    p.add_argument("--omega-span", type=float, default=240.0)
    p.add_argument("--omega-core-n", type=int, default=65)
    p.add_argument("--omega-tail-n", type=int, default=8)
    p.add_argument("--eps-min", type=float, default=-24.0)
    p.add_argument("--eps-max", type=float, default=160.0)
    p.add_argument("--n-eps", type=int, default=161)

    p.add_argument("--p-core-max", type=float, default=4.0)
    p.add_argument("--p-core-n", type=int, default=81)
    p.add_argument("--p-tail-n", type=int, default=41)
    p.add_argument("--angular-nodes", type=int, default=10)
    p.add_argument("--angular-feature-nodes", type=int, default=7)
    p.add_argument("--angular-chunk", type=int, default=4)
    p.add_argument("--coherent-nk", type=int, default=96)
    p.add_argument("--coherent-nquad", type=int, default=48)

    p.add_argument("--eta-gamma", type=float, default=0.002)
    p.add_argument(
        "--taper-mode",
        choices=("none", "qkin_cosine", "qkin_hard", "cosine", "hard"),
        default="qkin_cosine",
    )
    p.add_argument("--taper-start", type=float, default=0.0)
    p.add_argument("--taper-stop", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    up_path = args.up.expanduser().resolve()
    down_path = args.down.expanduser().resolve()
    for path in (up_path, down_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if (args.control_up is None) != (args.control_down is None):
        raise ValueError("--control-up and --control-down must be supplied together")

    cutoffs = _csv_floats(args.cutoffs)
    if cutoffs[0] <= 0.0:
        raise ValueError("cutoffs must be positive")

    eps0, mu_up, mu_dn = load_cube_physics(up_path)
    if not all(np.isfinite(x) for x in (eps0, mu_up, mu_dn)):
        raise ValueError("input cubes do not contain finite eps0, mu_up and mu_dn")
    with np.load(up_path) as data:
        mass = float(np.asarray(data["mass"]).reshape(-1)[0]) if "mass" in data.files else 1.0
    qff = abs(
        float(np.sqrt(max(mass * mu_up, 0.0)))
        - float(np.sqrt(max(mass * mu_dn, 0.0)))
    )
    if str(args.q).strip().lower() == "auto":
        q_values = np.unique(np.asarray([0.0, qff, qff + float(args.q_offset)], dtype=float))
    else:
        q_values = _csv_floats(args.q)
    if q_values[0] < 0.0:
        raise ValueError("Q probes must be non-negative")

    ku, wu, au_raw = load_akw_cube(up_path)
    kd, wd, ad_raw = load_akw_cube(down_path)
    au, up_desc = normalize_spectral_rows(
        au_raw,
        wu,
        load_cube_a_convention(up_path),
        enabled=False,
    )
    ad, down_desc = normalize_spectral_rows(
        ad_raw,
        wd,
        load_cube_a_convention(down_path),
        enabled=False,
    )

    can_rebuild = _npz_has(up_path, "ReS", "ImS") and _npz_has(down_path, "ReS", "ImS")
    line_mode = str(args.line_mode)
    rebuild = can_rebuild if line_mode == "auto" else line_mode == "rebuild"
    if rebuild and not can_rebuild:
        raise ValueError("--line-mode rebuild requested, but ReS/ImS are absent")

    omega = _omega_grid(
        q_values,
        mu_up,
        mu_dn,
        core=float(args.omega_core),
        span=float(args.omega_span),
        n_core=int(args.omega_core_n),
        n_tail=int(args.omega_tail_n),
    )
    up_w_min, up_w_max = _common_w_bounds(wu)
    eps_min = max(float(args.eps_min), up_w_min)
    eps_max = min(float(args.eps_max), up_w_max)
    eps = build_eps_grid(
        eps_min,
        eps_max,
        max(21, int(args.n_eps)),
        "split",
        2.5,
        include_points=omega,
    )

    safe_p_max = min(float(ku[-1]), float(kd[-1]) - float(np.max(q_values)))
    if float(cutoffs[-1]) > safe_p_max + 1.0e-12:
        raise ValueError(
            f"largest cutoff {cutoffs[-1]:g} exceeds safe cube support {safe_p_max:g} "
            f"for max Q={np.max(q_values):g}"
        )
    p_grid = _p_grid(
        cutoffs,
        q_values,
        mu_up,
        mu_dn,
        p_min=max(float(ku[0]), 0.0),
        core_max=float(args.p_core_max),
        n_core=int(args.p_core_n),
        n_tail=int(args.p_tail_n),
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] up={up_path}")
    print(f"[probe] down={down_path}")
    print(f"[probe] conventions: up [{up_desc}], down [{down_desc}]")
    print(f"[probe] mu=({mu_up:+.8g},{mu_dn:+.8g}), eps0={eps0:.8g}, QFFLO={qff:.8g}")
    print(
        f"[probe] Q={q_values.tolist()}, Lambda={cutoffs.tolist()}, "
        f"grids omega/p/eps={omega.size}/{p_grid.size}/{eps.size}, "
        f"line_mode={'rebuild' if rebuild else 'stored'}",
        flush=True,
    )

    # The same canonical QP model is used for the control cubes and the cheap
    # analytic add-back, matching fflo.pairbuild.
    model_up = qp_model_from_cube(up_path)
    model_down = qp_model_from_cube(down_path)

    with tempfile.TemporaryDirectory(prefix="bubble_cutoff_probe_") as tmp_raw:
        tmp = Path(tmp_raw)
        if args.control_up is None:
            control_up_path = tmp / "A0_up.npz"
            control_down_path = tmp / "A0_down.npz"
            build_qp_cube(up_path, control_up_path, model=model_up)
            build_qp_cube(down_path, control_down_path, model=model_down)
            print("[probe] generated canonical matched controls in a temporary directory")
        else:
            control_up_path = args.control_up.expanduser().resolve()
            control_down_path = args.control_down.expanduser().resolve()
            for path in (control_up_path, control_down_path):
                if not path.exists():
                    raise FileNotFoundError(path)

        control_ku, control_wu, control_au_raw = load_akw_cube(control_up_path)
        control_kd, control_wd, control_ad_raw = load_akw_cube(control_down_path)
        if not np.array_equal(ku, control_ku) or not np.array_equal(kd, control_kd):
            raise ValueError("full and control momentum grids differ")
        control_au, _ = normalize_spectral_rows(
            control_au_raw,
            control_wu,
            load_cube_a_convention(control_up_path),
            enabled=False,
        )
        control_ad, _ = normalize_spectral_rows(
            control_ad_raw,
            control_wd,
            load_cube_a_convention(control_down_path),
            enabled=False,
        )

        re_up = im_up = re_dn = im_dn = None
        sigma0_up = sigma0_dn = 0.0
        eta_up = eta_dn = 1.0e-3
        if rebuild:
            re_up, im_up, sigma0_up, eta_up = _load_sigma_rebuild_fields(up_path)
            re_dn, im_dn, sigma0_dn, eta_dn = _load_sigma_rebuild_fields(down_path)
            aup, up_support = _rebuild_a_bilinear(
                p_grid[:, None],
                eps[None, :],
                ku,
                wu,
                re_up,
                im_up,
                mu=mu_up,
                sigma0=sigma0_up,
                eta=eta_up,
            )
        else:
            aup, up_support = bilinear_eval_with_support(
                p_grid[:, None], eps[None, :], ku, wu, au
            )
        aup = np.where(up_support, aup, 0.0)
        control_up, control_up_support = bilinear_eval_with_support(
            p_grid[:, None], eps[None, :], control_ku, control_wu, control_au
        )
        control_up = np.where(control_up_support, control_up, 0.0)

        global _GRID_CTX
        _GRID_CTX = {
            "p": p_grid,
            "omega": omega,
            "eps": eps,
            "cutoffs": cutoffs,
            "kd": kd,
            "wd": wd,
            "ad": ad,
            "control_kd": control_kd,
            "control_wd": control_wd,
            "control_ad": control_ad,
            "aup": aup,
            "control_up": control_up,
            "re_dn": re_dn,
            "im_dn": im_dn,
            "sigma0_dn": sigma0_dn,
            "eta_dn": eta_dn,
            "mu_dn": mu_dn,
            "k_features": [np.sqrt(max(mu_up, 0.0)), np.sqrt(max(mu_dn, 0.0))],
            "angular_nodes": max(2, int(args.angular_nodes)),
            "angular_feature_nodes": max(3, int(args.angular_feature_nodes)),
            "angular_chunk": max(1, int(args.angular_chunk)),
        }

        residual_raw = np.zeros((cutoffs.size, q_values.size, omega.size), dtype=float)
        support_mean = np.zeros(q_values.size, dtype=float)
        tasks = list(enumerate(q_values.tolist()))
        workers = max(1, min(int(args.workers), len(tasks)))
        if workers == 1:
            rows = map(_grid_q_worker, tasks)
            for iq, row, support in rows:
                residual_raw[:, iq] = row
                support_mean[iq] = support
        else:
            # Cluster target is Linux; fork keeps the cube arrays copy-on-write
            # instead of serializing them once per Q row.
            with get_context("fork").Pool(processes=workers) as pool:
                for iq, row, support in pool.imap_unordered(_grid_q_worker, tasks, chunksize=1):
                    residual_raw[:, iq] = row
                    support_mean[iq] = support

    # pi * [raw/(2pi)^2] = raw/(4pi), exactly as finalize_impi_scan().
    im_grid_residual = residual_raw / (4.0 * np.pi)

    ku_qp, eu_qp, zu_qp, gu_qp, mu_up_qp = _canonical_qp_fields(up_path, model_up)
    kd_qp, ed_qp, zd_qp, gd_qp, mu_dn_qp = _canonical_qp_fields(down_path, model_down)
    qp_features = [_kf(ku_qp, eu_qp, mu_up_qp), _kf(kd_qp, ed_qp, mu_dn_qp)]
    im_qp = np.zeros_like(im_grid_residual)
    for il, cutoff in enumerate(cutoffs):
        _cv_init(
            omega,
            ku_qp, eu_qp, zu_qp, gu_qp,
            kd_qp, ed_qp, zd_qp, gd_qp,
            qp_features,
            {
                "angle_mode": "kjac",
                "nk": max(24, int(args.coherent_nk)),
                "nphi": 32,
                "n_kquad": max(8, int(args.coherent_nquad)),
                "kquad_chunk_size": 8,
                "kmax": float(cutoff),
                "gamma_floor": 1.0e-3,
            },
        )
        for iq, q_value in enumerate(q_values):
            im_qp[il, iq] = np.asarray(_cv_work(float(q_value)), dtype=float)
        print(f"[qp] Lambda={cutoff:g}: analytic add-back complete", flush=True)

    im_pi_num = im_grid_residual + im_qp
    im_pi_ref = np.zeros_like(im_pi_num)
    delta_im_raw = np.zeros_like(im_pi_num)
    re_delta_pi_zero = np.zeros((cutoffs.size, q_values.size), dtype=float)
    re_inv_gamma_zero = np.zeros_like(re_delta_pi_zero)
    io_zero = int(np.argmin(np.abs(omega)))
    if abs(float(omega[io_zero])) > 1.0e-12:
        raise RuntimeError("internal omega grid does not contain zero")

    for iq, q_value in enumerate(q_values):
        inv_ref = g0_inverse(
            omega,
            q=float(q_value),
            eps0=eps0,
            mu_up=mu_up,
            mu_dn=mu_dn,
            eta_gamma=float(args.eta_gamma),
        ) - proxy_medium_pi_from_gamma(
            omega,
            q=float(q_value),
            eps0=eps0,
            mu_up=mu_up,
            mu_dn=mu_dn,
            eta_gamma=float(args.eta_gamma),
        )
        inv_ref_zero = float(np.real(inv_ref[io_zero]))
        for il, cutoff in enumerate(cutoffs):
            pi_ref = proxy_full_pi_from_gamma(
                omega,
                q=float(q_value),
                eps0=eps0,
                mu_up=mu_up,
                mu_dn=mu_dn,
                eta_gamma=float(args.eta_gamma),
                cutoff=float(cutoff),
            )
            im_pi_ref[il, iq] = np.imag(pi_ref)
            raw = im_pi_num[il, iq] - im_pi_ref[il, iq]
            delta_im_raw[il, iq] = raw
            delta_re = kk_on_tapered_residual_native(
                omega,
                raw,
                q_value=float(q_value),
                taper_mode=str(args.taper_mode),
                taper_start=float(args.taper_start),
                taper_stop=float(args.taper_stop),
                mu_up=mu_up,
                mu_dn=mu_dn,
            )
            re_delta_pi_zero[il, iq] = float(delta_re[io_zero])
            re_inv_gamma_zero[il, iq] = inv_ref_zero - float(delta_re[io_zero])

    reference = re_inv_gamma_zero[-1]
    shift_to_lmax = re_inv_gamma_zero - reference[None, :]
    max_abs_shift = np.max(np.abs(shift_to_lmax), axis=1)
    next_step = np.full(cutoffs.size, np.nan, dtype=float)
    if cutoffs.size > 1:
        next_step[:-1] = np.max(
            np.abs(re_inv_gamma_zero[:-1] - re_inv_gamma_zero[1:]), axis=1
        )

    print("\n# QUICK CUTOFF SATURATION ESTIMATE")
    print("# shift_to_Lmax is the estimated change in Re Gamma^{-1}(Q,0)")
    print(f"# reference Lambda_max={cutoffs[-1]:g}; support_mean={support_mean.tolist()}")
    print("Lambda   max|shift_to_Lmax|   max|step_to_next|")
    for il, cutoff in enumerate(cutoffs):
        step_text = "-" if not np.isfinite(next_step[il]) else f"{next_step[il]:.8e}"
        print(f"{cutoff:6.2f}   {max_abs_shift[il]:.8e}       {step_text}")

    tsv_path = out_dir / "cutoff_probe.tsv"
    with tsv_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "Lambda\tQ\tReDeltaPi0\tReInvGamma0\tshift_to_Lmax\t"
            "max_abs_delta_im\tsupport_mean\n"
        )
        for il, cutoff in enumerate(cutoffs):
            for iq, q_value in enumerate(q_values):
                stream.write(
                    f"{cutoff:.12g}\t{q_value:.12g}\t"
                    f"{re_delta_pi_zero[il, iq]:+.12e}\t"
                    f"{re_inv_gamma_zero[il, iq]:+.12e}\t"
                    f"{shift_to_lmax[il, iq]:+.12e}\t"
                    f"{np.max(np.abs(delta_im_raw[il, iq])):.12e}\t"
                    f"{support_mean[iq]:.12e}\n"
                )

    npz_path = out_dir / "cutoff_probe.npz"
    np.savez_compressed(
        npz_path,
        cutoff=cutoffs,
        q=q_values,
        omega=omega,
        p=p_grid,
        eps=eps,
        ImPiGridResidual=im_grid_residual,
        ImPiQPQP=im_qp,
        ImPiNum=im_pi_num,
        ImPiReference=im_pi_ref,
        ImDeltaPiRaw=delta_im_raw,
        ReDeltaPiZero=re_delta_pi_zero,
        ReInvGammaZero=re_inv_gamma_zero,
        ShiftToLambdaMax=shift_to_lmax,
        MaxAbsShiftToLambdaMax=max_abs_shift,
        MaxAbsStepToNext=next_step,
        support_mean=support_mean,
        eps0=np.asarray(eps0),
        mu_up=np.asarray(mu_up),
        mu_dn=np.asarray(mu_dn),
        qff=np.asarray(qff),
        eta_gamma=np.asarray(float(args.eta_gamma)),
        taper_mode=np.asarray(str(args.taper_mode)),
        taper_start=np.asarray(float(args.taper_start)),
        taper_stop=np.asarray(float(args.taper_stop)),
        line_mode=np.asarray("rebuild" if rebuild else "stored"),
        up_path=np.asarray(str(up_path)),
        down_path=np.asarray(str(down_path)),
    )
    print(f"\n[probe] wrote {tsv_path}")
    print(f"[probe] wrote {npz_path}")


if __name__ == "__main__":
    main()
