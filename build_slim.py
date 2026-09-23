#!/usr/bin/env python3
"""Build a slim copy of the FFLO pipeline containing only the code the live loop
can actually reach.

WHY THIS IS A SCRIPT AND NOT A ONE-OFF EDIT
-------------------------------------------
The source tree is never modified, so the legacy runners keep working and this is
reversible.  Re-run it after any change upstream and the slim copy is regenerated.
Nothing here is hand-trimmed: every deletion is the output of a reachability
analysis, and the header of each emitted file lists exactly what was dropped.

THE ALGORITHM
-------------
A fixed point over (module -> set of names required from it):

  1. seed with the four entry points and the name `main`;
  2. for a module M with required set R, compute the top-level symbols reachable
     from R by walking the AST (a bare name, an attribute, a callback passed by
     reference or a mention at module level all count as a use);
  3. for every `from X import a, b` that survives inside that reachable code, add
     a, b to the requirements of X, and pull X into the closure;
  4. repeat until nothing changes.

The reachability step is deliberately conservative, so it UNDER-reports dead code:
whatever it drops really is unreachable.  What it cannot see is dynamic dispatch by
string (getattr, a name looked up from a config table).  The verification pass at
the end therefore imports every emitted module and checks that every required name
is present -- and the numeric gate (compare one density step against the original)
is what actually licenses using this for physics.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "X_FFLO_FULLGRID_PROD"))
PKG = os.path.join(HERE, "fflo")

STDLIB_OK = {
    "numpy", "scipy", "os", "sys", "math", "argparse", "json", "time", "pathlib",
    "typing", "dataclasses", "functools", "itertools", "collections", "concurrent",
    "warnings", "subprocess", "hashlib", "shutil", "textwrap", "copy", "re",
    "traceback", "contextlib", "__future__", "datetime", "multiprocessing",
    "tempfile", "pickle", "gc", "platform", "random", "string", "io", "csv",
    "matplotlib", "glob", "logging", "inspect", "types", "enum", "abc", "uuid",
    "operator", "bisect", "heapq", "statistics", "signal", "errno", "stat",
}

# source file  ->  module name inside the slim package
LAYOUT = {
    "diagnostics/density_from_clean_pair_one_shot.py":        "density",
    "diagnostics/run_inline_qp_product_p065.py":              "pairbuild",
    "true_akw_pipeline/make_impi_table.py":                   "impi_table",
    "true_akw_pipeline/make_pair_gamma_table_proxy_residual.py": "pair_gamma",
    "true_akw_pipeline/run_lorentzian_self_consistency_fixed_mu.py": "sigma_engine",
    "true_akw_pipeline/compute_convolution_from_adn_table.py": "kramers_kronig",
    "true_akw_pipeline/compute_sigma_from_pair_convolution.py": "sigma_from_pair",
    "true_akw_pipeline/make_impi_analytic_cv.py":             "impi_cv",
    "true_akw_pipeline/analytic_qp_bubble.py":                "analytic_bubble",
    "true_akw_pipeline/build_qp_cube.py":                     "qp_cube",
    "true_akw_pipeline/qp_dyson_extract.py":                  "qp_dyson",
    "true_akw_pipeline/make_true_akw_cubes_nomega_grid.py":   "spectral",
    "true_akw_pipeline/make_lorentzian_akw_cubes.py":         "lorentzian_cube",
    "true_akw_pipeline/precompute_adn_phi_table.py":          "adn_phi",
    "true_akw_pipeline/lattice_grids.py":                     "lattice_grids",
    "true_akw_pipeline/decompose_akw_qp_inc.py":              "decompose_qp_inc",
    "true_akw_pipeline/make_true_akw_cubes_feature_grid_poleaware.py": "feature_grid",
    "critical_line/pair_shifted.py":                          "pair_shifted",
    "FFLO_exec/proxy.py":                                     "proxy",
    "simga_test.py":                                          "sigma_test",
    "diagnostics/compare_sigma_from_pair_motors.py":          "compare_motors",
}
# basename (as written in an import) -> source path
BY_BASENAME = {os.path.basename(p)[:-3]: p for p in LAYOUT}
BY_BASENAME["FFLO_exec.proxy"] = "FFLO_exec/proxy.py"

# Literal fixes applied after trimming, so the slim tree is self-contained.
# Each one is here rather than hand-edited into fflo/, so that re-running the
# build reproduces the package exactly.
PATCHES = {
    "pairbuild": [
        # the two downstream stages are invoked as subprocesses; in the source tree
        # they are file paths under true_akw_pipeline/, here they are modules
        ('            str(PIPELINE / "make_impi_table.py"),',
         '            "-m", "fflo.impi_table",'),
        ('            str(PIPELINE / "make_pair_gamma_table_proxy_residual.py"),',
         '            "-m", "fflo.pair_gamma",'),
        # ROOT is already the slim root (parents[1] of fflo/pairbuild.py); PIPELINE
        # pointed at a directory that no longer exists
        ('PIPELINE = ROOT / "true_akw_pipeline"\nsys.path.insert(0, str(PIPELINE))',
         'PIPELINE = ROOT\nsys.path.insert(0, str(ROOT))'),
    ],
}

ENTRIES = {
    "diagnostics/density_from_clean_pair_one_shot.py": {"main"},
    "diagnostics/run_inline_qp_product_p065.py": {"main"},
    "true_akw_pipeline/make_impi_table.py": {"main"},
    "true_akw_pipeline/make_pair_gamma_table_proxy_residual.py": {"main"},
}


def span(node):
    """Line range of a top-level definition, INCLUDING its decorators.

    ast puts `node.lineno` on the `def`/`class` keyword, not on the `@`.  Deleting
    by `node.lineno` therefore leaves the decorators behind, where they silently
    re-attach to whatever definition comes next -- which showed up here as
    `@dataclass(frozen=True)` landing on a function and raising
    "'function' object has no attribute '__mro__'" at import time.
    """
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(start, min(d.lineno for d in node.decorator_list))
    return start, node.end_lineno


def top_symbols(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = n
    return out


def is_main_guard(n):
    return (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name) and n.test.left.id == "__name__")


def names_in(node, universe, exclude=None):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in universe:
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in universe:
            out.add(sub.attr)
    if exclude:
        out.discard(exclude)
    return out


def reachable(tree, required, keep_main_guard):
    """Top-level symbols reachable from `required` plus module-level execution."""
    tops = top_symbols(tree)
    universe = set(tops)
    refs = {name: names_in(node, universe, exclude=name) for name, node in tops.items()}

    module_level = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if is_main_guard(n) and not keep_main_guard:
            continue
        module_level |= names_in(n, universe)

    start = (set(required) & universe) | module_level
    seen, q = set(start), deque(start)
    while q:
        for nxt in refs.get(q.popleft(), ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return tops, seen


def collect_imports(tree, tops, live, keep_main_guard):
    """(module basename -> names) for imports that survive inside live code."""
    wanted = {}

    def note(n):
        if isinstance(n, ast.ImportFrom) and n.module:
            base = n.module.split(".")[-1]
            key = n.module if n.module in BY_BASENAME else base
            if key in BY_BASENAME:
                wanted.setdefault(key, set()).update(
                    a.name for a in n.names if a.name != "*")
        elif isinstance(n, ast.Import):
            for a in n.names:
                key = a.name if a.name in BY_BASENAME else a.name.split(".")[0]
                if key in BY_BASENAME:
                    wanted.setdefault(key, set())

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name in live:
                for sub in ast.walk(n):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        note(sub)
            continue
        if is_main_guard(n) and not keep_main_guard:
            continue
        for sub in ast.walk(n):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                note(sub)
    return wanted


def solve_closure():
    required = {p: set(s) for p, s in ENTRIES.items()}
    trees = {}
    changed = True
    while changed:
        changed = False
        for path in list(required):
            full = os.path.join(SRC, path)
            if path not in trees:
                trees[path] = ast.parse(open(full).read())
            tree = trees[path]
            tops, live = reachable(tree, required[path], keep_main_guard=path in ENTRIES)
            for base, names in collect_imports(tree, tops, live, path in ENTRIES).items():
                tgt = BY_BASENAME[base]
                before = set(required.get(tgt, set()))
                required.setdefault(tgt, set()).update(names)
                if tgt not in trees or required[tgt] != before:
                    changed = True
    return required, trees


def rewrite_imports(src_text):
    """Point every intra-project import at the flat `fflo.` namespace.

    Handles both shapes the source tree uses:
        from true_akw_pipeline.foo import bar   ->  from fflo.foo import bar
        from foo import bar                     ->  from fflo.foo import bar
        import foo as f                         ->  from fflo import foo as f
        import foo                              ->  from fflo import foo as foo
    The `import X as Y` shape is the one that bites: the source imports
    `simga_test as st` (note the typo in the original filename) and a rewriter that
    only looked at `from ... import ...` would leave it dangling.
    """
    PREFIXES = ("true_akw_pipeline.", "diagnostics.", "critical_line.", "FFLO_exec.")
    out = []
    for line in src_text.splitlines(keepends=True):
        indent = line[: len(line) - len(line.lstrip())]
        s = line.strip()

        if s.startswith("from "):
            head = s.split(" import ", 1)
            if len(head) == 2:
                modpath = head[0][len("from "):].strip()
                key = modpath
                for pre in PREFIXES:
                    if key.startswith(pre):
                        key = key[len(pre):]
                if modpath in BY_BASENAME:
                    key = modpath
                if key in BY_BASENAME:
                    mod = LAYOUT[BY_BASENAME[key]]
                    line = f"{indent}from fflo.{mod} import {head[1]}\n"

        elif s.startswith("import "):
            body = s[len("import "):]
            if "," not in body:                       # one module per statement
                parts = body.split(" as ")
                dotted = parts[0].strip()
                alias = parts[1].strip() if len(parts) > 1 else None
                key = dotted
                for pre in PREFIXES:
                    if key.startswith(pre):
                        key = key[len(pre):]
                if dotted in BY_BASENAME:
                    key = dotted
                if key in BY_BASENAME:
                    mod = LAYOUT[BY_BASENAME[key]]
                    name = alias or dotted.split(".")[-1]
                    line = f"{indent}from fflo import {mod} as {name}\n"

        out.append(line)
    return "".join(out)


def emit(path, required, tree):
    full = os.path.join(SRC, path)
    lines = open(full).read().splitlines(keepends=True)
    is_entry = path in ENTRIES
    tops, live = reachable(tree, required, keep_main_guard=is_entry)
    dead = sorted(set(tops) - live, key=lambda s: span(tops[s])[0])

    drop = set()
    for s in dead:
        a, b = span(tops[s])
        drop.update(range(a, b + 1))
    if not is_entry:
        for n in tree.body:
            if is_main_guard(n):
                drop.update(range(n.lineno, n.end_lineno + 1))

    dead_lines = len(drop)
    kept = "".join(ln for i, ln in enumerate(lines, 1) if i not in drop)
    kept = rewrite_imports(kept)
    for old, new in PATCHES.get(LAYOUT[path], ()):
        if old not in kept:
            raise SystemExit(
                f"patch non applicabile in {LAYOUT[path]}: il sorgente e' cambiato.\n"
                f"  cercavo: {old!r}")
        kept = kept.replace(old, new)

    header = [
        "# =====================================================================\n",
        f"# SLIM copy of {path}\n",
        "#\n",
        f"# {len(dead)} top-level symbols ({dead_lines} lines, "
        f"{100*dead_lines//max(len(lines),1)}% of the original) removed as\n",
        "# unreachable from the live pipeline.  Required entry points here:\n",
        f"#   {', '.join(sorted(required)) if required else '(module-level only)'}\n",
        "#\n",
        "# Generated by build_slim.py -- do not edit by hand, edit the source in\n",
        f"# {os.path.relpath(SRC, HERE)}/ and re-run the build.\n",
    ]
    if dead:
        header.append("#\n# Removed:\n")
        header += [f"#   {s}\n" for s in dead]
    header.append("# =====================================================================\n")

    dst = os.path.join(PKG, LAYOUT[path] + ".py")
    os.makedirs(PKG, exist_ok=True)
    open(dst, "w").write("".join(header) + kept)
    return len(lines), dead_lines, len(dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="wipe fflo/ first")
    a = ap.parse_args()
    if a.clean and os.path.isdir(PKG):
        shutil.rmtree(PKG)

    required, trees = solve_closure()
    print(f"chiusura: {len(required)} moduli\n")
    print(f"{'modulo':24s} {'orig':>7s} {'tolte':>7s} {'resta':>7s}  {'%':>4s}")
    print("-" * 56)
    tot_o = tot_d = 0
    for path in sorted(required, key=lambda p: LAYOUT[p]):
        if path not in trees:
            trees[path] = ast.parse(open(os.path.join(SRC, path)).read())
        o, d, nsym = emit(path, required[path], trees[path])
        tot_o += o
        tot_d += d
        print(f"{LAYOUT[path]:24s} {o:7d} {d:7d} {o-d:7d}  {100*d//max(o,1):3d}%")
    print("-" * 56)
    print(f"{'TOTALE':24s} {tot_o:7d} {tot_d:7d} {tot_o-tot_d:7d}  "
          f"{100*tot_d//max(tot_o,1):3d}%")

    open(os.path.join(PKG, "__init__.py"), "w").write(
        '"""Slim FFLO pipeline -- generated by build_slim.py, do not edit."""\n')

    # ---- verification ---------------------------------------------------
    # Importing is a much stronger check than compiling: it runs every module-level
    # statement, so a decorator left dangling or a name removed from under a
    # reference fails here rather than in the middle of a run.
    print("\nverifica: import di ogni modulo + presenza dei simboli richiesti")
    sys.path.insert(0, HERE)
    import importlib
    bad = 0
    for path in sorted(required, key=lambda p: LAYOUT[p]):
        mod = LAYOUT[path]
        try:
            m = importlib.import_module(f"fflo.{mod}")
        except Exception as e:
            print(f"  IMPORT FALLITO  fflo.{mod}: {type(e).__name__}: {e}")
            bad += 1
            continue
        missing = [n for n in sorted(required[path]) if not hasattr(m, n)]
        if missing:
            print(f"  SIMBOLI MANCANTI fflo.{mod}: {missing}")
            bad += 1
    print("  tutto verde" if not bad else f"  {bad} moduli con problemi")
    if bad:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
