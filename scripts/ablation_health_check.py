#!/usr/bin/env python3
"""Read-only pre-scale health check for the TRAP ablation study.

Before launching a large-scale ablation run this script reports, per target
model and experiment, whether the next run will REUSE a cached result or
RE-RUN it (and *why*), plus whether the existing on-disk outputs are complete.
It mirrors the exact cache-matching logic in ``main.run_ablation_mode`` /
``_load_cached_ablation_result`` and the study_id / parameter resolution done by
``framework_orchestrator.step_35_run_ablation`` + ``run_ours_ablation``.

It writes NOTHING. In particular it never calls ``prepare_kb_snapshot`` (which
copies the live KB); instead it reads the hash of the already-frozen
``initial_snapshot.pkl`` when present, falling back to hashing the live snapshot
source read-only.

Usage (from repo root or anywhere):
    python -m scripts.ablation_health_check
    python -m scripts.ablation_health_check --config path/to/config.env
    python -m scripts.ablation_health_check --json     # machine-readable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve repo layout regardless of cwd.
_OURS_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _OURS_ROOT
for _p in (str(_REPO_ROOT), str(_OURS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.ablation_manifest import AblationPaths  # noqa: E402
from utils.experiment_config import (  # noqa: E402
    load_experiment_registry,
    resolve_kb_parameters,
    resolve_seeds,
    resolve_snapshot_path,
)

# Policy version stamped into D's cache key (mirror of main._RAG_ONLY_POLICY_VERSION).
# Imported lazily to avoid importing the heavy main module at start-up.
def _rag_only_policy_version() -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ours_main_probe", str(_OURS_ROOT / "main.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "_RAG_ONLY_POLICY_VERSION", "")


# Files every completed run must contain, relative to the run root.
_REQUIRED_RUN_FILES = [
    ("raw", "result.json"),
    ("raw", "behaviors.jsonl"),
    ("raw", "manifest.json"),
    ("reports", "rl_learning_curve.csv"),
]


def _load_config_env(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        cfg[key.strip()] = val.split("#")[0].strip()
    return cfg


def _slugify_model_name(model_name: str) -> str:
    """Match framework_orchestrator._slugify_model_name exactly."""
    slug = model_name.lower().replace("/", "__").replace(":", "_")
    slug = re.sub(r"[^a-z0-9_\-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "default"


def _resolve_targets(cfg: Dict[str, str]) -> List[str]:
    multi = (cfg.get("ENABLE_MULTI_TARGET", "false") or "false").lower() == "true"
    models_str = (cfg.get("TARGET_MODELS_LIST", "") or "").strip()
    targets = [m.strip() for m in models_str.split(",") if m.strip()] if models_str else []
    if not targets:
        single = cfg.get("TARGET_MODEL", "").strip()
        targets = [single] if single else []
    # When multi-target the orchestrator suffixes study_id per model; otherwise not.
    return targets, multi


def _resolve_evaluator(cfg: Dict[str, str]) -> str:
    for key in ("UNIFIED_JUDGE_MODEL", "JUDGE_MODEL", "EVAL_MODEL"):
        v = (cfg.get(key) or "").strip()
        if v:
            return v
    return ""


def _resolve_n_behaviors(cfg: Dict[str, str]) -> Tuple[int, str]:
    goals_file = (cfg.get("OURS_ABLATION_GOALS_FILE", "") or "").strip()
    if goals_file:
        gp = Path(goals_file)
        if not gp.is_absolute():
            gp = _OURS_ROOT / goals_file
        if gp.exists():
            try:
                data = json.loads(gp.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return len(data), f"goals_file={gp.name}"
            except Exception:
                pass
        return -1, f"goals_file={goals_file} (unreadable)"
    start = int(cfg.get("START_INDEX", "0") or "0")
    end = int(cfg.get("END_INDEX", "10") or "10")
    return max(end - start, 0), f"dataset slice {start}..{end}"


def _snapshot_sha(run_paths: AblationPaths, live_source: Optional[Path], rag_enabled: bool) -> str:
    """Return the sha256 the next run WILL use, without writing anything.

    With the idempotent prepare_kb_snapshot, a frozen initial_snapshot.pkl is
    reused verbatim; so if it exists we hash that. Otherwise the next run will
    freeze the live source, so we hash that instead (read-only).
    """
    if not rag_enabled:
        return ""
    frozen = run_paths.kb / "initial_snapshot.pkl"
    target = frozen if frozen.exists() else (live_source if live_source and live_source.exists() else None)
    if target is None:
        return ""
    return hashlib.sha256(Path(target).read_bytes()).hexdigest()


def _empirical_seconds_per_iter(default: float = 60.0) -> Tuple[float, int]:
    """Median wall-seconds per iteration, learned from existing ablation runs.

    Uses the median (not mean) so a single pathological 5-hour run does not
    inflate the estimate. Returns (seconds_per_iter, n_runs_sampled). Falls back
    to ``default`` when no prior runs exist. This signal predates the cheaper
    beam=1 / fewer-refine settings, so it is a conservative (high) estimate.
    """
    import statistics

    root = _OURS_ROOT / "results" / "ablations"
    samples: List[float] = []
    for rj in root.glob("*/*/*/raw/result.json"):
        try:
            d = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        elapsed = float(d.get("elapsed_seconds", 0) or 0)
        n_iters = sum(len(b.get("iterations", []) or []) for b in d.get("behaviors", []) or [])
        if elapsed > 0 and n_iters > 0:
            samples.append(elapsed / n_iters)
    if not samples:
        return default, 0
    return statistics.median(samples), len(samples)


def _estimate_rerun_seconds(
    n_behaviors: int,
    hard_cap_iters: int,
    sec_per_iter: float,
) -> Tuple[float, float]:
    """(optimistic, pessimistic) wall-seconds to re-run one experiment.

    optimistic  — most behaviors jailbreak within a few iterations (~3).
    pessimistic — every behavior exhausts the hard iteration cap (no success).
    Reality usually lands between the two; report both so planning is honest.
    """
    if n_behaviors <= 0:
        return 0.0, 0.0
    optimistic = n_behaviors * min(3, hard_cap_iters) * sec_per_iter
    pessimistic = n_behaviors * hard_cap_iters * sec_per_iter
    return optimistic, pessimistic


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _check_completeness(run_paths: AblationPaths) -> List[str]:
    issues: List[str] = []
    for sub, name in _REQUIRED_RUN_FILES:
        p = (run_paths.raw if sub == "raw" else run_paths.reports) / name
        if not p.exists():
            issues.append(f"missing {sub}/{name}")
        elif p.stat().st_size == 0:
            issues.append(f"empty {sub}/{name}")
    # result.json must parse and carry behaviors.
    rj = run_paths.result_json_path()
    if rj.exists() and rj.stat().st_size > 0:
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
            behs = data.get("behaviors")
            if not isinstance(behs, list) or not behs:
                issues.append("result.json has no behaviors")
        except Exception as exc:
            issues.append(f"result.json unparseable: {exc}")
    return issues


def _cache_mismatch_reason(manifest: Dict[str, Any], expected: Dict[str, Any]) -> Optional[str]:
    for key, value in expected.items():
        if manifest.get(key) != value:
            return f"{key}: cached={manifest.get(key)!r} != expected={value!r}"
    return None


def check_study(
    study_id: str,
    target_model: str,
    evaluator_model: str,
    n_behaviors: int,
    max_iterations: int,
    registry: Dict[str, Any],
    experiments_filter: Optional[List[str]],
    policy_version: str,
    seed_override: Optional[int] = None,
    hard_cap_iters: int = 50,
    sec_per_iter: float = 60.0,
) -> Dict[str, Any]:
    shared = registry.get("shared", {})
    experiments = registry.get("experiments", {})
    seeds = resolve_seeds(registry, seed_override)
    snapshot_source = resolve_snapshot_path(registry, None)
    selected = {
        k: v for k, v in experiments.items()
        if not experiments_filter or k in experiments_filter
    }

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for exp_id, exp_cfg in sorted(selected.items()):
            rag_enabled = bool(exp_cfg.get("rag_enabled", True))
            tree_variant = exp_cfg.get("tree_variant", "full_active")
            rag_mode = str(exp_cfg.get("rag_mode", "online" if rag_enabled else "disabled")).lower()
            run_kb_params = resolve_kb_parameters(registry, exp_cfg)
            if str(tree_variant).lower() == "none":
                run_kb_params = {**run_kb_params, "rag_only_policy_version": policy_version}

            paths = AblationPaths(study_id, exp_id, seed)
            manifest_p = paths.manifest_path()
            result_p = paths.result_json_path()

            snap_sha = _snapshot_sha(paths, snapshot_source if rag_enabled else None, rag_enabled)

            expected = {
                "study_id": study_id,
                "experiment_id": exp_id,
                "experiment_name": exp_cfg.get("name", exp_id),
                "seed": seed,
                "target_model": target_model,
                "evaluator_model": evaluator_model,
                "rag_enabled": rag_enabled,
                "tree_variant": tree_variant,
                "tree_leaves": exp_cfg.get("tree_leaves"),
                "tree_max_depth": exp_cfg.get("tree_max_depth"),
                "success_threshold": shared.get("success_threshold", 7.0),
                "max_iterations_per_episode": max_iterations,
                "episode_budget": n_behaviors,
                "rag_mode": rag_mode,
                "kb_snapshot_sha256": snap_sha,
                "kb_parameters": run_kb_params,
            }

            decision = "RE-RUN"
            reason = ""
            completeness: List[str] = []
            if not manifest_p.exists() or not result_p.exists():
                reason = "no cached run on disk"
            else:
                try:
                    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
                except Exception as exc:
                    reason = f"manifest unreadable: {exc}"
                    manifest = None
                if manifest is not None:
                    mismatch = _cache_mismatch_reason(manifest, expected)
                    if mismatch is None:
                        decision = "REUSE"
                        reason = "all cache keys match"
                        completeness = _check_completeness(paths)
                    else:
                        reason = mismatch

            # Only RE-RUN rows cost time; REUSE rows are free.
            if decision == "RE-RUN":
                est_lo, est_hi = _estimate_rerun_seconds(n_behaviors, hard_cap_iters, sec_per_iter)
            else:
                est_lo, est_hi = 0.0, 0.0

            rows.append({
                "experiment_id": exp_id,
                "seed": seed,
                "decision": decision,
                "reason": reason,
                "completeness_issues": completeness,
                "est_seconds_optimistic": round(est_lo, 1),
                "est_seconds_pessimistic": round(est_hi, 1),
            })
    return {"study_id": study_id, "target_model": target_model, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only ablation cache/completeness health check")
    ap.add_argument("--config", default=str(_REPO_ROOT / "config.env"), help="Path to config.env")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    cfg = _load_config_env(Path(args.config))
    registry = load_experiment_registry(_OURS_ROOT / "config" / "experiment_registry.yaml")

    base_study_id = cfg.get("OURS_ABLATION_STUDY_ID", "study_main")
    targets, multi = _resolve_targets(cfg)
    evaluator = _resolve_evaluator(cfg)
    n_behaviors, n_src = _resolve_n_behaviors(cfg)
    max_iterations = int(cfg.get("OURS_MAX_ITERATIONS", "50") or "50")
    # The loop stops a behavior at the hard cap; the effective per-behavior
    # iteration ceiling is min(MAX_ITERATIONS, MAX_ITER_HARD_CAP).
    hard_cap = int(cfg.get("OURS_MAX_ITER_HARD_CAP", str(max_iterations)) or max_iterations)
    eff_cap = min(max_iterations, hard_cap)
    exp_str = (cfg.get("OURS_ABLATION_EXPERIMENTS", "") or "").strip()
    experiments_filter = [e.strip() for e in exp_str.split(",") if e.strip()] or None
    policy_version = _rag_only_policy_version()
    sec_per_iter, n_timing_samples = _empirical_seconds_per_iter()

    # Mirror run_ours_ablation: a single-seed override pins the ablation to one
    # seed (config.env keeps this at 1 to avoid 5x repeated seed runs).
    seed_override: Optional[int] = None
    for seed_key in ("OURS_ABLATION_RANDOM_SEED", "OURS_SEED", "RANDOM_SEED"):
        raw_seed = (cfg.get(seed_key) or "").strip()
        if raw_seed:
            try:
                seed_override = int(raw_seed)
                break
            except ValueError:
                pass

    if not targets:
        print("[health-check] No target model configured (TARGET_MODEL / TARGET_MODELS_LIST empty).")
        return 2

    studies: List[Dict[str, Any]] = []
    for tm in targets:
        study_id = f"{base_study_id}__{_slugify_model_name(tm)}" if (multi and len(targets) > 1) else base_study_id
        studies.append(check_study(
            study_id=study_id, target_model=tm, evaluator_model=evaluator,
            n_behaviors=n_behaviors, max_iterations=max_iterations,
            registry=registry, experiments_filter=experiments_filter,
            policy_version=policy_version, seed_override=seed_override,
            hard_cap_iters=eff_cap, sec_per_iter=sec_per_iter,
        ))

    grand_lo = sum(r["est_seconds_optimistic"] for s in studies for r in s["rows"])
    grand_hi = sum(r["est_seconds_pessimistic"] for s in studies for r in s["rows"])

    if args.json:
        print(json.dumps({
            "base_study_id": base_study_id,
            "targets": targets,
            "evaluator_model": evaluator,
            "n_behaviors": n_behaviors,
            "n_behaviors_source": n_src,
            "max_iterations": max_iterations,
            "effective_iter_cap": eff_cap,
            "experiments_filter": experiments_filter,
            "rag_only_policy_version": policy_version,
            "timing": {
                "seconds_per_iter": round(sec_per_iter, 1),
                "timing_samples": n_timing_samples,
                "est_total_seconds_optimistic": round(grand_lo, 1),
                "est_total_seconds_pessimistic": round(grand_hi, 1),
            },
            "studies": studies,
        }, indent=2, ensure_ascii=False))
        return 0

    print("=" * 78)
    print("OURS ABLATION — PRE-SCALE HEALTH CHECK (read-only)")
    print("=" * 78)
    print(f"config            : {args.config}")
    print(f"base study_id     : {base_study_id}")
    print(f"targets           : {len(targets)}  (multi_target_suffix={'yes' if multi and len(targets)>1 else 'no'})")
    print(f"evaluator_model   : {evaluator or '(unset)'}")
    print(f"behaviors / study : {n_behaviors}  [{n_src}]")
    print(f"max_iterations    : {max_iterations}  (effective cap {eff_cap})")
    print(f"experiments       : {experiments_filter or 'all'}")
    print(f"D policy version  : {policy_version}")
    if n_timing_samples:
        print(f"timing model      : {sec_per_iter:.0f}s/iter (median of {n_timing_samples} prior runs; "
              f"pre-dates beam=1, so conservative)")
    else:
        print(f"timing model      : {sec_per_iter:.0f}s/iter (default — no prior runs to learn from)")
    print()

    total_reuse = total_rerun = total_incomplete = 0
    for study in studies:
        print("-" * 78)
        print(f"TARGET  {study['target_model']}")
        print(f"STUDY   {study['study_id']}")
        study_lo = study_hi = 0.0
        for row in study["rows"]:
            tag = "REUSE " if row["decision"] == "REUSE" else "RE-RUN"
            line = f"  [{tag}] {row['experiment_id']:9} seed={row['seed']:<5} {row['reason']}"
            print(line)
            if row["decision"] == "REUSE":
                total_reuse += 1
                if row["completeness_issues"]:
                    total_incomplete += 1
                    for iss in row["completeness_issues"]:
                        print(f"            ! INCOMPLETE: {iss}")
            else:
                total_rerun += 1
                study_lo += row["est_seconds_optimistic"]
                study_hi += row["est_seconds_pessimistic"]
        if study_hi > 0:
            print(f"  est. re-run time for this target: {_fmt_hms(study_lo)} – {_fmt_hms(study_hi)}")
    print("-" * 78)
    print(f"SUMMARY: reuse={total_reuse}  re-run={total_rerun}  "
          f"reused-but-incomplete={total_incomplete}")
    print(f"ESTIMATED TOTAL RE-RUN TIME: {_fmt_hms(grand_lo)} (optimistic) "
          f"– {_fmt_hms(grand_hi)} (pessimistic)")
    print(f"  optimistic = most behaviors jailbreak within ~3 iters; "
          f"pessimistic = every behavior hits the {eff_cap}-iter cap.")
    print(f"  NOTE: serial estimate; parallel execution (MAX_PARALLEL_*) divides wall-time roughly by worker count.")
    if total_incomplete:
        print("WARNING: some cached runs that would be REUSED are incomplete/corrupt — "
              "delete those run dirs to force a clean re-run before scaling.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
