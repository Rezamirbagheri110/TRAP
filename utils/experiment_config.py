"""Canonical experiment-registry accessors.

All entrypoints should resolve seeds, KB parameters, and experiment variants
through this module so environment defaults cannot silently drift by script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

OURS_ROOT = Path(__file__).parent.parent
DEFAULT_REGISTRY_PATH = OURS_ROOT / "config" / "experiment_registry.yaml"


def load_experiment_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    registry_path = path or DEFAULT_REGISTRY_PATH
    with registry_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment registry must be a mapping: {registry_path}")
    return data


def resolve_seeds(registry: Dict[str, Any], override: Optional[int] = None) -> List[int]:
    if override is not None:
        return [int(override)]
    shared = registry.get("shared", {})
    seeds = shared.get("seeds") or [shared.get("random_seed", 42)]
    return [int(seed) for seed in seeds]


def resolve_kb_parameters(
    registry: Dict[str, Any],
    experiment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    shared = dict(registry.get("shared", {}).get("kb_parameters") or {})
    return {**shared, **dict((experiment or {}).get("kb_parameters") or {})}


def resolve_snapshot_path(
    registry: Dict[str, Any],
    override: Optional[str] = None,
) -> Optional[Path]:
    raw = override or registry.get("shared", {}).get("kb_snapshot")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else OURS_ROOT / path
