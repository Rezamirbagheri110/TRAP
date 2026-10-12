"""
Tree schema validator — runs against action_tree_schema.yaml and the live ActionTree.

Checks:
  1. No duplicate leaf names across the entire tree
  2. Max depth constraint per variant (full_active=4, pruned=2, flat=1)
  3. Leaf counts match expected values from schema meta
  4. Disabled categories are absent from active leaf set
  5. All pruned_leaves exist in the live ActionTree
  6. Migration map old_names are gone from the live ActionTree
  7. YAML is parseable and has required top-level keys
  8. Experiment registry counts are consistent with schema counts
  9. Pruned leaf category mapping is coherent with canonical tree paths

Exit code 0 = all checks passed.
Exit code 1 = one or more failures (printed to stderr).

Usage:
    py -3 helpers/validate_tree_schema.py
    py -3 helpers/validate_tree_schema.py --schema config/action_tree_schema.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

# Force UTF-8 output on Windows so unicode symbols print correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Optional yaml import — give a clear error if missing
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Locate project root and make imports work regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.action_tree import ActionTree  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_leaves_from_yaml(nodes: list, depth: int = 1,
                               disabled_roots: Set[str] = None,
                               _parent_disabled: bool = False) -> List[Tuple[str, int, bool]]:
    """Recursively collect (name, depth, is_disabled) for every leaf in YAML nodes."""
    if disabled_roots is None:
        disabled_roots = set()
    results = []
    for node in nodes:
        name = node.get("name", "")
        is_disabled = _parent_disabled or node.get("disabled", False) or (name in disabled_roots)
        children = node.get("children", [])
        if node.get("leaf") or not children:
            results.append((name, depth, is_disabled))
        else:
            results.extend(
                _collect_leaves_from_yaml(
                    children, depth + 1,
                    disabled_roots | ({name} if is_disabled else set()),
                    _parent_disabled=is_disabled,
                )
            )
    return results


def _collect_active_leaves(nodes: list, disabled_roots: Set[str]) -> List[Tuple[str, int]]:
    """Return (name, depth) for leaves that are NOT under a disabled root."""
    all_leaves = _collect_leaves_from_yaml(nodes, depth=1, disabled_roots=disabled_roots)
    return [(n, d) for n, d, dis in all_leaves if not dis]


def _collect_leaf_names(node: dict) -> Set[str]:
    """Collect leaf names under a node recursively."""
    names: Set[str] = set()
    children = node.get("children", [])
    if node.get("leaf") or not children:
        name = node.get("name")
        if name:
            names.add(name)
        return names
    for child in children:
        names |= _collect_leaf_names(child)
    return names


def _live_leaf_names(tree: ActionTree) -> Set[str]:
    return {n.name for n in tree.leaf_nodes}


def _filtered_experiment_view(experiment: Dict[str, Any], ignored_keys: Set[str]) -> Dict[str, Any]:
    """Return a shallow experiment view excluding descriptive keys."""
    return {
        key: value for key, value in experiment.items()
        if key not in ignored_keys
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self):
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def fail(self, msg: str):
        self.failures.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0


def check_yaml_structure(schema: dict, result: ValidationResult):
    required_keys = {"meta", "attack_tree", "auxiliary_tree", "pruned_leaves", "migration_map"}
    missing = required_keys - set(schema.keys())
    if missing:
        result.fail(f"YAML missing top-level keys: {missing}")


def check_no_duplicate_leaves(schema: dict, result: ValidationResult):
    nodes = schema.get("attack_tree", [])
    disabled = {n["name"] for n in nodes if n.get("disabled")}
    all_leaves = _collect_leaves_from_yaml(nodes)
    names = [n for n, d, dis in all_leaves]
    seen: Dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        result.fail(f"Duplicate leaf names in attack_tree YAML: {dupes}")


def check_max_depth(schema: dict, result: ValidationResult):
    meta = schema.get("meta", {})
    expected_max = meta.get("expected_counts", {}).get("max_depth_full", 3)
    nodes = schema.get("attack_tree", [])
    disabled = {n["name"] for n in nodes if n.get("disabled")}
    active = _collect_active_leaves(nodes, disabled)
    violations = [(n, d) for n, d in active if d > expected_max]
    if violations:
        result.fail(
            f"Leaves exceed max_depth_full={expected_max}: "
            + ", ".join(f"{n}(depth={d})" for n, d in violations[:10])
        )


def check_leaf_counts(schema: dict, result: ValidationResult):
    meta = schema.get("meta", {}).get("expected_counts", {})
    nodes = schema.get("attack_tree", [])

    # pass top-level disabled root names so propagation works correctly
    disabled_roots = {n["name"] for n in nodes if n.get("disabled")}
    all_leaves = _collect_leaves_from_yaml(nodes, disabled_roots=disabled_roots)
    active_leaves = [(n, d) for n, d, dis in all_leaves if not dis]
    disabled_leaves = [(n, d) for n, d, dis in all_leaves if dis]

    actual_active = len(active_leaves)
    actual_total  = len(all_leaves)
    expected_active = meta.get("full_active_leaves")
    expected_total  = meta.get("full_total_leaves")

    if expected_active is not None and actual_active != expected_active:
        result.fail(
            f"Active leaf count mismatch: schema says {expected_active}, "
            f"YAML has {actual_active}"
        )
    if expected_total is not None and actual_total != expected_total:
        result.fail(
            f"Total leaf count (incl. disabled): schema says {expected_total}, "
            f"YAML has {actual_total}"
        )

    # pruned_leaves count
    pruned = schema.get("pruned_leaves", {})
    flat_list = [leaf for cat_leaves in pruned.values() for leaf in cat_leaves]
    expected_pruned = meta.get("pruned_leaves", 36)
    if len(flat_list) != expected_pruned:
        result.fail(
            f"Pruned leaf count mismatch: expected {expected_pruned}, "
            f"got {len(flat_list)}"
        )


def check_disabled_absent_from_active(schema: dict, result: ValidationResult):
    meta = schema.get("meta", {})
    disabled_cats = set(meta.get("text_only_policy", {}).get("disabled_categories", []))
    nodes = schema.get("attack_tree", [])
    for node in nodes:
        if node.get("name") in disabled_cats and not node.get("disabled"):
            result.fail(
                f"Category '{node['name']}' is in text_only_policy.disabled_categories "
                f"but not marked disabled: true in attack_tree"
            )


def check_pruned_leaves_exist_in_live_tree(schema: dict, tree: ActionTree,
                                           result: ValidationResult):
    pruned = schema.get("pruned_leaves", {})
    flat_list = [leaf for cat_leaves in pruned.values() for leaf in cat_leaves]
    live = _live_leaf_names(tree)
    missing = [name for name in flat_list if name not in live]
    if missing:
        result.fail(
            f"Pruned leaves not found in live ActionTree: {missing}"
        )


def check_migration_old_names_gone(schema: dict, tree: ActionTree,
                                   result: ValidationResult):
    migration = schema.get("migration_map", {})
    live = _live_leaf_names(tree)
    live_all = set(tree.node_index.keys())
    for old_name, entry in migration.items():
        status = entry.get("status")
        if status in ("renamed", "removed"):
            if old_name in live_all:
                result.fail(
                    f"Migration map says '{old_name}' was {status}, "
                    f"but it still exists in the live tree"
                )


def check_no_duplicate_in_live_tree(tree: ActionTree, result: ValidationResult):
    names = [n.name for n in tree.leaf_nodes]
    seen: Dict[str, int] = {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        result.fail(f"Duplicate leaf names in live ActionTree: {dupes}")


def check_live_counts_match_schema(schema: dict, tree_full: ActionTree,
                                   tree_full_active: ActionTree,
                                   result: ValidationResult):
    meta = schema.get("meta", {}).get("expected_counts", {})
    expected_total = meta.get("full_total_leaves")
    expected_active = meta.get("full_active_leaves")
    actual_total = len(tree_full.leaf_nodes)
    actual_active = len(tree_full_active.leaf_nodes)
    if expected_total is not None and actual_total != expected_total:
        result.fail(
            f"Live ActionTree(full) has {actual_total} leaves; "
            f"schema meta.full_total_leaves={expected_total}"
        )
    if expected_active is not None and actual_active != expected_active:
        result.fail(
            f"Live ActionTree(full_active) has {actual_active} leaves; "
            f"schema meta.full_active_leaves={expected_active}"
        )


def check_registry_consistency(schema: dict, result: ValidationResult):
    registry_path = PROJECT_ROOT / "config" / "experiment_registry.yaml"
    if not registry_path.exists():
        result.fail(f"Experiment registry not found: {registry_path}")
        return

    with open(registry_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    expected = schema.get("meta", {}).get("expected_counts", {})
    expected_active = expected.get("full_active_leaves")
    expected_pruned = expected.get("pruned_leaves")
    schema_version = schema.get("meta", {}).get("version")

    shared = registry.get("shared", {})
    experiments = registry.get("experiments", {})

    if str(shared.get("schema_version")) != str(schema_version):
        result.fail(
            f"Registry shared.schema_version={shared.get('schema_version')} "
            f"!= schema.meta.version={schema_version}"
        )

    a = experiments.get("A", {})
    b = experiments.get("B", {})
    c = experiments.get("C", {})
    d = experiments.get("D", {})
    e = experiments.get("E", {})

    if expected_active is not None and a.get("tree_leaves") != expected_active:
        result.fail(f"Experiment A tree_leaves={a.get('tree_leaves')} != schema full_active_leaves={expected_active}")
    if expected_pruned is not None and b.get("tree_leaves") != expected_pruned:
        result.fail(f"Experiment B tree_leaves={b.get('tree_leaves')} != schema pruned_leaves={expected_pruned}")
    if expected_pruned is not None and c.get("tree_leaves") != expected_pruned:
        result.fail(f"Experiment C tree_leaves={c.get('tree_leaves')} != schema pruned_leaves={expected_pruned}")
    # D (Retrieval-Driven) has no tree: tree_variant=none and tree_leaves must be null.
    if d.get("tree_variant") != "none":
        result.fail(f"Experiment D tree_variant={d.get('tree_variant')} != expected 'none'")
    if d.get("tree_leaves") is not None:
        result.fail(f"Experiment D tree_leaves={d.get('tree_leaves')} != expected null (no tree)")
    # E (Full+NoRAG) must mirror A exactly except rag_enabled=False.
    # This is the paired comparison for RAG contribution (A↔E).
    if not experiments.get("E"):
        result.fail("Experiment E is missing from registry (required for rag_effect comparison A↔E)")
    else:
        if e.get("tree_variant") != a.get("tree_variant"):
            result.fail(
                f"Experiment E tree_variant={e.get('tree_variant')} != A tree_variant={a.get('tree_variant')}; "
                "E must mirror A to isolate RAG contribution"
            )
        if expected_active is not None and e.get("tree_leaves") != expected_active:
            result.fail(
                f"Experiment E tree_leaves={e.get('tree_leaves')} != schema full_active_leaves={expected_active}; "
                "E must use the same leaf count as A"
            )
        if e.get("rag_enabled") is not False:
            result.fail(
                f"Experiment E rag_enabled={e.get('rag_enabled')} — must be false to isolate RAG contribution"
            )
        if a.get("rag_enabled") is not True:
            result.fail(
                f"Experiment A rag_enabled={a.get('rag_enabled')} — must be true for A↔E to be a valid comparison"
            )


def check_registry_pair_invariants(schema: dict, result: ValidationResult):
    """Enforce pairwise ablation invariants not covered by simple count checks."""
    registry_path = PROJECT_ROOT / "config" / "experiment_registry.yaml"
    if not registry_path.exists():
        result.fail(f"Experiment registry not found: {registry_path}")
        return

    with open(registry_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    experiments = registry.get("experiments", {})
    a = experiments.get("A", {}) or {}
    e = experiments.get("E", {}) or {}
    if not a or not e:
        return

    ignored_for_pair = {
        "name", "description", "measures", "interpretation_note",
        "rag_enabled", "rag_mode",
    }
    a_view = _filtered_experiment_view(a, ignored_for_pair)
    e_view = _filtered_experiment_view(e, ignored_for_pair)
    if a_view != e_view:
        differing_keys = sorted(
            key for key in (set(a_view) | set(e_view))
            if a_view.get(key) != e_view.get(key)
        )
        result.fail(
            "Experiment E must mirror A for all non-descriptive fields except RAG mode/enabled; "
            f"differing keys: {differing_keys}"
        )


def check_pruned_category_mapping(schema: dict, result: ValidationResult):
    """Ensure pruned_leaves categories map to intended canonical pools.

    v1.3 pruned categories are semantic buckets for ablation, not always strict
    canonical subtrees. So we validate against explicit allowed pools.
    """
    attack_nodes = schema.get("attack_tree", [])
    by_name = {n.get("name"): n for n in attack_nodes}

    # Build allowed sets for pruned category keys
    allowed: Dict[str, Set[str]] = {}
    if "jailbreak_techniques" in by_name:
        allowed["jailbreak_techniques"] = _collect_leaf_names(by_name["jailbreak_techniques"])
        # pruned jailbreak bucket intentionally includes two fine_tuning leaves.
        if "modern_ai_techniques" in by_name:
            mat = by_name["modern_ai_techniques"]
            ft = next((c for c in mat.get("children", []) if c.get("name") == "fine_tuning_behavior"), None)
            if ft:
                allowed["jailbreak_techniques"] |= _collect_leaf_names(ft)
    if "roleplay_simulations" in by_name:
        allowed["roleplay_simulations"] = _collect_leaf_names(by_name["roleplay_simulations"])
        # pruned roleplay bucket includes story/chaos wrappers by design.
        if "jailbreak_techniques" in by_name:
            jb = by_name["jailbreak_techniques"]
            narrative = next((c for c in jb.get("children", []) if c.get("name") == "narrative"), None)
            if narrative:
                allowed["roleplay_simulations"] |= _collect_leaf_names(narrative)
        if "chaos_custom" in by_name:
            allowed["roleplay_simulations"] |= _collect_leaf_names(by_name["chaos_custom"])
        # legacy leaf retained in pruned set for continuity
        allowed["roleplay_simulations"].add("roleplay_scenario")
    if "social_engineering" in by_name:
        allowed["social_engineering"] = _collect_leaf_names(by_name["social_engineering"])
    if "steganography_obfuscation" in by_name:
        allowed["steganography_obfuscation"] = _collect_leaf_names(by_name["steganography_obfuscation"])
        # leetspeak is intentionally reused from encoding_transforms.
        if "jailbreak_techniques" in by_name:
            jb = by_name["jailbreak_techniques"]
            enc = next((c for c in jb.get("children", []) if c.get("name") == "encoding_transforms"), None)
            if enc:
                allowed["steganography_obfuscation"] |= _collect_leaf_names(enc)

    # Aliases used in pruned_leaves
    if "modern_ai_techniques" in by_name:
        mat = by_name["modern_ai_techniques"]
        prompt_inj = next((c for c in mat.get("children", []) if c.get("name") == "prompt_injection"), None)
        if prompt_inj:
            allowed["prompt_injection"] = _collect_leaf_names(prompt_inj)
    if "linguistic_semantic_attacks" in by_name:
        allowed["linguistic_semantic"] = _collect_leaf_names(by_name["linguistic_semantic_attacks"])

    pruned = schema.get("pruned_leaves", {})
    for key, leaves in pruned.items():
        if key not in allowed:
            result.fail(f"pruned_leaves category '{key}' has no canonical mapping in attack_tree")
            continue
        invalid = [leaf for leaf in leaves if leaf not in allowed[key]]
        if invalid:
            result.fail(
                f"pruned_leaves category '{key}' has leaves outside canonical path: {invalid}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate action tree schema against live tree")
    parser.add_argument(
        "--schema",
        default=str(PROJECT_ROOT / "config" / "action_tree_schema.yaml"),
        help="Path to action_tree_schema.yaml"
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}", file=sys.stderr)
        sys.exit(1)

    with open(schema_path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    print(f"Schema: {schema_path}")
    print(f"Version: {schema.get('meta', {}).get('version', '?')}  "
          f"Status: {schema.get('meta', {}).get('status', '?')}")
    print()

    result = ValidationResult()

    print("Loading live ActionTree...")
    tree = ActionTree(tree_variant="full")
    tree_full_active = ActionTree(tree_variant="full_active")
    tree_pruned = ActionTree(tree_variant="pruned")
    print(f"  Live leaves (full): {len(tree.leaf_nodes)}")
    print(f"  Live leaves (full_active): {len(tree_full_active.leaf_nodes)}")
    print(f"  Live leaves (pruned): {len(tree_pruned.leaf_nodes)}")
    print()

    result = ValidationResult()
    checks = [
        ("YAML structure",              lambda: check_yaml_structure(schema, result)),
        ("No duplicate leaves (YAML)",  lambda: check_no_duplicate_leaves(schema, result)),
        ("Max depth (YAML)",            lambda: check_max_depth(schema, result)),
        ("Leaf counts (YAML)",          lambda: check_leaf_counts(schema, result)),
        ("Disabled policy",             lambda: check_disabled_absent_from_active(schema, result)),
        ("Pruned leaves in live tree",  lambda: check_pruned_leaves_exist_in_live_tree(schema, tree_pruned, result)),
        ("Migration: old names gone",   lambda: check_migration_old_names_gone(schema, tree, result)),
        ("No duplicate leaves (live)",  lambda: check_no_duplicate_in_live_tree(tree, result)),
        ("Live counts vs schema",       lambda: check_live_counts_match_schema(schema, tree, tree_full_active, result)),
        ("Registry consistency",        lambda: check_registry_consistency(schema, result)),
        ("Registry pair invariants",    lambda: check_registry_pair_invariants(schema, result)),
        ("Pruned category mapping",     lambda: check_pruned_category_mapping(schema, result)),
    ]

    print(f"{'Check':<45} {'Result'}")
    print("-" * 60)
    for label, check_fn in checks:
        before_f = len(result.failures)
        before_w = len(result.warnings)
        check_fn()
        new_f = len(result.failures) - before_f
        new_w = len(result.warnings) - before_w
        if new_f:
            tag = "FAIL"
        elif new_w:
            tag = "WARN"
        else:
            tag = "ok"
        print(f"  {label:<43} {tag}")

    print()
    if result.failures:
        print(f"FAILURES ({len(result.failures)}):")
        for f in result.failures:
            print(f"  ✗ {f}", file=sys.stderr)
            print(f"  ✗ {f}")
    if result.warnings:
        print(f"WARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  ⚠ {w}")

    print()
    if result.ok:
        print("All checks passed.")
        sys.exit(0)
    else:
        print(f"{len(result.failures)} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
