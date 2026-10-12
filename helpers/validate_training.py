"""
Post-run diagnostic script for gradient-bandit training logs.
Usage: python helpers/validate_training.py --log /path/to/rl_attack.log

The policy is a gradient bandit (Sutton & Barto §2.8), NOT an actor-critic, so
there is no policy_loss / value_loss / advantage to inspect. The meaningful
health signals are:

  - policy entropy: should be > 0 and should generally trend DOWN as the policy
    concentrates on good branches (entropy collapsing to ~0 immediately = no
    exploration; staying flat at max = no learning).
  - avg batch reward: should be finite and ideally trend UP over the run.

These are parsed from the "[TRAIN] Bandit metrics: ..." log lines emitted by
run_attack_with_rag.py.
"""

import argparse
import re
import sys
from pathlib import Path


def validate_log_file(log_path: str) -> bool:
    """Validate that gradient-bandit training diagnostics are healthy."""

    print("=" * 80)
    print("GRADIENT-BANDIT TRAINING DIAGNOSTICS VALIDATION")
    print("=" * 80)

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
    except Exception as e:
        print(f"[FAIL] Error reading log file: {e}")
        return False

    entropies = [float(m.group(1)) for m in re.finditer(r"entropy=([-\d.]+)", log_content)]
    avg_rewards = [float(m.group(1)) for m in re.finditer(r"avg_reward=([-\d.]+)", log_content)]

    issues = []
    warnings = []

    print("\n1. Policy Entropy:")
    if not entropies:
        issues.append("[FAIL] No entropy values found (no bandit train steps logged!)")
    else:
        first, last = entropies[0], entropies[-1]
        print(f"   Count: {len(entropies)}")
        print(f"   Range: {min(entropies):.4f} - {max(entropies):.4f}")
        print(f"   First -> Last: {first:.4f} -> {last:.4f}")
        if max(entropies) <= 1e-6:
            issues.append("[FAIL] Entropy is ~0 throughout (policy never explored)")
        elif last > first:
            warnings.append("[WARN] Entropy increased over the run (policy diffusing, not concentrating)")
        else:
            print("   [OK] Entropy positive and trending toward concentration")

    print("\n2. Average Batch Reward:")
    if not avg_rewards:
        warnings.append("[WARN] No avg_reward values found")
    else:
        first, last = avg_rewards[0], avg_rewards[-1]
        print(f"   Count: {len(avg_rewards)}")
        print(f"   Range: {min(avg_rewards):.4f} - {max(avg_rewards):.4f}")
        print(f"   First -> Last: {first:.4f} -> {last:.4f}")
        if any(r != r for r in avg_rewards):  # NaN check
            issues.append("[FAIL] avg_reward contains NaN")
        elif last < first:
            warnings.append("[WARN] avg_reward decreased over the run")
        else:
            print("   [OK] avg_reward finite and non-decreasing")

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if issues:
        print("\n[CRITICAL ISSUES]:")
        for issue in issues:
            print(f"  {issue}")
        print("\n[FAIL] TRAINING LOG CHECK FAILED")
        return False

    if warnings:
        print("\n[WARNINGS]:")
        for warning in warnings:
            print(f"  {warning}")
        print("\n[WARN] VALIDATION PASSED WITH WARNINGS - Proceed with caution")
        return True

    print("\n[SUCCESS] TRAINING LOG CHECKS PASSED")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect training diagnostics from an existing run log.")
    parser.add_argument("--log", type=Path, default=Path(__file__).resolve().parent.parent / "logs" / "rl_attack.log", help="Path to an existing gradient-bandit run log")
    log_file = parser.parse_args().log

    if not log_file.exists():
        print(f"No run log found: {log_file}. Run an experiment first or provide --log; this is a post-run diagnostic.")
        sys.exit(1)

    success = validate_log_file(str(log_file))
    sys.exit(0 if success else 1)
