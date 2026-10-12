"""
Ablation study completeness validator.

Runs fail-fast checks before any downstream analysis to catch missing runs,
wrong seeds, incomplete artifacts, or schema violations early.

Exit codes:
  0  — all checks passed
  1  — one or more checks failed (details printed to stdout)

Usage:
    python scripts/ablation_validator.py --study <study_id>
    python scripts/ablation_validator.py --study <study_id> --seeds 42 123 --models deepseek mistral
    python scripts/ablation_validator.py --study <study_id> --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_OURS_DIR = Path(__file__).parent.parent
if str(_OURS_DIR) not in sys.path:
    sys.path.insert(0, str(_OURS_DIR))

import yaml

from utils.ablation_manifest import _RESULTS_ROOT, load_manifest
from utils.schemas import SCHEMA_VERSION
from utils.behavior_aggregate import extract_behaviors_from_result

_REGISTRY_PATH = _OURS_DIR / "config" / "experiment_registry.yaml"

REQUIRED_RAW_ARTIFACTS = ["manifest.json", "result.json", "behaviors.jsonl"]
REQUIRED_REPORT_ARTIFACTS = ["behaviors.csv", "summary.md"]
REQUIRED_AGGREGATE_ARTIFACTS = ["behaviors_all.csv", "ablation_summary.csv"]


# ---------------------------------------------------------------------------
# Check result helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(f"[FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(f"[WARN] {msg}")

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        for w in self.warnings:
            print(w)
        for e in self.errors:
            print(e)
        if self.ok:
            print(f"[OK] All checks passed ({len(self.warnings)} warnings)")
        else:
            print(f"\n[SUMMARY] {len(self.errors)} error(s), {len(self.warnings)} warning(s)")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_experiments_present(
    study_dir: Path,
    expected_experiments: Set[str],
    cr: CheckResult,
) -> Dict[str, List[Path]]:
    """Check that each expected experiment has at least one run directory."""
    found: Dict[str, List[Path]] = {}
    for exp_id in expected_experiments:
        exp_dir = study_dir / exp_id
        if not exp_dir.exists():
            cr.fail(f"Experiment '{exp_id}' directory missing: {exp_dir}")
            found[exp_id] = []
            continue
        seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        if not seed_dirs:
            cr.fail(f"Experiment '{exp_id}' has no seed runs in {exp_dir}")
        found[exp_id] = seed_dirs
    return found


def check_seeds_present(
    found: Dict[str, List[Path]],
    expected_seeds: Set[int],
    cr: CheckResult,
) -> None:
    """Check that every expected seed directory exists for each experiment."""
    for exp_id, seed_dirs in found.items():
        existing_seeds = {int(d.name) for d in seed_dirs if d.name.isdigit()}
        for seed in expected_seeds:
            if seed not in existing_seeds:
                cr.fail(f"Experiment '{exp_id}': seed {seed} missing (found: {sorted(existing_seeds)})")


def check_raw_artifacts(
    found: Dict[str, List[Path]],
    cr: CheckResult,
) -> None:
    """Check that required raw/ files exist in every run directory."""
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            raw_dir = seed_dir / "raw"
            if not raw_dir.exists():
                cr.fail(f"{exp_id}/{seed_dir.name}: raw/ directory missing")
                continue
            for artifact in REQUIRED_RAW_ARTIFACTS:
                artifact_path = raw_dir / artifact
                if not artifact_path.exists():
                    cr.fail(f"{exp_id}/{seed_dir.name}/raw/{artifact}: missing")
                elif artifact_path.stat().st_size == 0:
                    cr.fail(f"{exp_id}/{seed_dir.name}/raw/{artifact}: empty")


def check_report_artifacts(
    found: Dict[str, List[Path]],
    cr: CheckResult,
    strict: bool = False,
) -> None:
    """Check report/ artifacts. In strict mode, treat missing as errors."""
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            reports_dir = seed_dir / "reports"
            if not reports_dir.exists():
                (cr.fail if strict else cr.warn)(
                    f"{exp_id}/{seed_dir.name}: reports/ directory missing"
                )
                continue
            for artifact in REQUIRED_REPORT_ARTIFACTS:
                artifact_path = reports_dir / artifact
                if not artifact_path.exists():
                    (cr.fail if strict else cr.warn)(
                        f"{exp_id}/{seed_dir.name}/reports/{artifact}: missing"
                    )
                elif artifact_path.stat().st_size == 0:
                    (cr.fail if strict else cr.warn)(
                        f"{exp_id}/{seed_dir.name}/reports/{artifact}: empty"
                    )


def check_aggregate_artifacts(
    study_dir: Path,
    cr: CheckResult,
    strict: bool = False,
) -> None:
    """Check that aggregate/ outputs exist."""
    agg_dir = study_dir / "aggregate"
    if not agg_dir.exists():
        (cr.fail if strict else cr.warn)(f"aggregate/ directory missing in {study_dir}")
        return
    for artifact in REQUIRED_AGGREGATE_ARTIFACTS:
        artifact_path = agg_dir / artifact
        if not artifact_path.exists():
            (cr.fail if strict else cr.warn)(f"aggregate/{artifact}: missing")
        elif artifact_path.stat().st_size == 0:
            (cr.fail if strict else cr.warn)(f"aggregate/{artifact}: empty")


def check_manifest_schema(
    found: Dict[str, List[Path]],
    cr: CheckResult,
) -> None:
    """Check manifest.json files for required fields and schema version."""
    required_fields = [
        "schema_version", "study_id", "experiment_id", "experiment_name",
        "seed", "timestamp", "git_commit", "rag_enabled", "tree_variant",
    ]
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            manifest_path = seed_dir / "raw" / "manifest.json"
            if not manifest_path.exists():
                continue  # already caught by check_raw_artifacts
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                cr.fail(f"{exp_id}/{seed_dir.name}/raw/manifest.json: JSON parse error — {exc}")
                continue

            for field in required_fields:
                if field not in manifest:
                    cr.fail(
                        f"{exp_id}/{seed_dir.name}/raw/manifest.json: missing field '{field}'"
                    )

            schema_ver = manifest.get("schema_version", "")
            if schema_ver != SCHEMA_VERSION:
                cr.warn(
                    f"{exp_id}/{seed_dir.name}/raw/manifest.json: "
                    f"schema_version={schema_ver!r}, expected {SCHEMA_VERSION!r}"
                )

            if not manifest.get("git_commit"):
                cr.warn(
                    f"{exp_id}/{seed_dir.name}/raw/manifest.json: git_commit is empty "
                    f"(run may not be reproducible)"
                )

            if manifest.get("git_dirty"):
                cr.warn(
                    f"{exp_id}/{seed_dir.name}/raw/manifest.json: git_dirty=True "
                    f"(working tree was dirty when this run was recorded)"
                )


def check_result_json_schema(
    found: Dict[str, List[Path]],
    cr: CheckResult,
    min_behaviors: int = 1,
) -> None:
    """Check that result.json files are loadable and contain behavior data."""
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            result_path = seed_dir / "raw" / "result.json"
            if not result_path.exists():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                cr.fail(f"{exp_id}/{seed_dir.name}/raw/result.json: JSON parse error — {exc}")
                continue

            behaviors = extract_behaviors_from_result(result)
            if len(behaviors) < min_behaviors:
                cr.fail(
                    f"{exp_id}/{seed_dir.name}/raw/result.json: "
                    f"found {len(behaviors)} behaviors, "
                    f"expected >= {min_behaviors}"
                )


def check_target_models_consistent(
    found: Dict[str, List[Path]],
    expected_models: Optional[Set[str]],
    cr: CheckResult,
) -> None:
    """Warn if target_model differs across experiments (makes comparison unreliable)."""
    if not expected_models:
        return
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            manifest_path = seed_dir / "raw" / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            model = manifest.get("target_model", "")
            if expected_models and model not in expected_models:
                cr.warn(
                    f"{exp_id}/{seed_dir.name}: target_model={model!r} not in "
                    f"expected set {expected_models}"
                )


def check_kb_isolation(
    found: Dict[str, List[Path]],
    cr: CheckResult,
) -> None:
    """Ensure RAG runs declare one immutable starting snapshot and controlled parameters."""
    snapshot_hashes: Set[str] = set()
    parameter_fingerprints: Set[str] = set()
    for exp_id, seed_dirs in found.items():
        for seed_dir in seed_dirs:
            manifest_path = seed_dir / "raw" / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not manifest.get("rag_enabled"):
                continue
            mode = manifest.get("rag_mode")
            if mode not in {"static", "online"}:
                cr.fail(f"{exp_id}/{seed_dir.name}: invalid or missing rag_mode={mode!r}")
            digest = str(manifest.get("kb_snapshot_sha256") or "")
            if not digest:
                cr.fail(f"{exp_id}/{seed_dir.name}: missing kb_snapshot_sha256")
            else:
                snapshot_hashes.add(digest)
            params = manifest.get("kb_parameters")
            if not isinstance(params, dict):
                cr.fail(f"{exp_id}/{seed_dir.name}: missing kb_parameters mapping")
            else:
                parameter_fingerprints.add(json.dumps(params, sort_keys=True, default=str))
    if len(snapshot_hashes) > 1:
        cr.fail(f"RAG runs use different initial KB snapshot hashes: {sorted(snapshot_hashes)}")
    if len(parameter_fingerprints) > 1:
        cr.fail("RAG runs use different KB parameters; comparison is confounded")


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_study(
    study_id: str,
    registry_path: Path = _REGISTRY_PATH,
    expected_seeds: Optional[Set[int]] = None,
    expected_models: Optional[Set[str]] = None,
    strict: bool = False,
    min_behaviors: int = 1,
) -> CheckResult:
    cr = CheckResult()

    with registry_path.open(encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    experiments: Dict[str, Any] = registry.get("experiments", {})
    expected_experiments = set(experiments.keys())
    if expected_seeds is None:
        shared = registry.get("shared", {})
        expected_seeds = set(shared.get("seeds") or [])
        if not expected_seeds and shared.get("random_seed") is not None:
            expected_seeds = {shared["random_seed"]}

    study_dir = _RESULTS_ROOT / study_id
    print(f"Validating study: {study_dir}")
    print(f"Expected experiments: {sorted(expected_experiments)}")
    print(f"Expected seeds: {sorted(expected_seeds)}")
    print()

    found = check_experiments_present(study_dir, expected_experiments, cr)
    check_seeds_present(found, expected_seeds, cr)
    check_raw_artifacts(found, cr)
    check_report_artifacts(found, cr, strict=strict)
    check_aggregate_artifacts(study_dir, cr, strict=strict)
    check_manifest_schema(found, cr)
    check_result_json_schema(found, cr, min_behaviors=min_behaviors)
    check_target_models_consistent(found, expected_models, cr)
    check_kb_isolation(found, cr)

    return cr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ablation study completeness")
    parser.add_argument("--study", required=True, help="Study ID")
    parser.add_argument("--registry", default=str(_REGISTRY_PATH))
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Expected seeds (default: from registry shared.random_seed)")
    parser.add_argument("--models", nargs="+",
                        help="Expected target model names (optional)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat missing report artifacts as errors (not warnings)")
    parser.add_argument("--min-behaviors", type=int, default=1,
                        help="Minimum behaviors expected per result.json")
    args = parser.parse_args()

    cr = validate_study(
        study_id=args.study,
        registry_path=Path(args.registry),
        expected_seeds=set(args.seeds) if args.seeds else None,
        expected_models=set(args.models) if args.models else None,
        strict=args.strict,
        min_behaviors=args.min_behaviors,
    )

    cr.print_report()
    sys.exit(0 if cr.ok else 1)


if __name__ == "__main__":
    main()
