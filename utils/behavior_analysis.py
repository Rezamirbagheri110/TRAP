"""
Behavior analysis utilities for the RAG-enhanced attack runner.

Tracks iteration outcomes by behavior, template effectiveness, and RL
metrics so the orchestrator can emit diagnostics at the end of each run.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BehaviorAnalyzer:
    """Builds behavioral summaries and template comparisons across attack runs."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.behavior_records: Dict[str, List[Dict[str, Any]]] = {}
        self.template_summary: Dict[str, Dict[str, Any]] = {}
        self.total_records = 0

    def record_iteration(self, goal: str, info: Dict[str, Any]) -> None:
        """Record a single attack iteration outcome."""
        normalized_goal = (goal or "unknown").strip()
        if not normalized_goal:
            normalized_goal = "unknown"
        record = info.copy()
        record['goal'] = normalized_goal
        self.behavior_records.setdefault(normalized_goal, []).append(record)
        self.total_records += 1

        template_name = record.get('template_name') or "unknown"
        template_entry = self.template_summary.setdefault(template_name, {
            'uses': 0,
            'successes': 0,
            'scores': [],
            'avg_iterations_to_success': [],
            'behavior_ids': set(),
            'first_used_timestamp': datetime.now().isoformat(),
            'rl_metrics': {}
        })
        template_entry['uses'] += 1
        template_entry['scores'].append(record.get('score', 0.0))
        template_entry['behavior_ids'].add(normalized_goal)
        if record.get('success'):
            template_entry['successes'] += 1
            iteration_value = record.get('iteration')
            if isinstance(iteration_value, (int, float)):
                template_entry['avg_iterations_to_success'].append(iteration_value)

        rl_metrics = record.get('rl_metrics') or {}
        for metric_name, raw_value in rl_metrics.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            bucket = 'success' if record.get('success') else 'failure'
            metric_entry = template_entry['rl_metrics'].setdefault(metric_name, {'success': [], 'failure': []})
            metric_entry[bucket].append(value)

    def analyze(self) -> Dict[str, Any]:
        """Produce consolidated behavior and template summaries."""
        behavior_summary = {}
        for goal, records in self.behavior_records.items():
            total = len(records)
            successes = sum(1 for record in records if record.get('success'))
            template_counter: Dict[str, int] = defaultdict(int)
            for record in records:
                template_key = record.get('template_name') or "unknown"
                template_counter[template_key] += 1
            behavior_summary[goal] = {
                'total_attempts': total,
                'successes': successes,
                'failures': total - successes,
                'success_rate': successes / total if total else 0.0,
                'average_score': sum(record.get('score', 0.0) for record in records) / total if total else 0.0,
                'templates': dict(template_counter),
                'rl_metrics': self._aggregate_record_rl_metrics(records)
            }

        template_insights = []
        for template_name, stats in self.template_summary.items():
            uses = stats.get('uses', 0)
            successes = stats.get('successes', 0)
            behavior_ids = sorted(stats.get('behavior_ids', []))
            iterations = stats.get('avg_iterations_to_success', [])
            avg_iterations = (sum(iterations) / len(iterations)) if iterations else None
            scores = stats.get('scores', [])
            avg_score = sum(scores) / len(scores) if scores else 0.0
            template_insights.append({
                'template_name': template_name,
                'uses': uses,
                'successes': successes,
                'success_rate': successes / uses if uses else 0.0,
                'avg_score': avg_score,
                'avg_iterations_to_success': avg_iterations,
                'behavior_ids': behavior_ids,
                'behavior_count': len(behavior_ids),
                'first_used_timestamp': stats.get('first_used_timestamp'),
                'rl_metrics': self._aggregate_template_rl_metrics(stats.get('rl_metrics', {}))
            })

        ranked_templates = sorted(
            template_insights,
            key=lambda entry: (entry['success_rate'], entry['avg_score']),
            reverse=True
        )

        return {
            'generated_at': datetime.now().isoformat(),
            'total_records': self.total_records,
            'behaviors': behavior_summary,
            'templates': ranked_templates,
            'top_templates': ranked_templates[:5]
        }

    def save_analysis(self, path: Path) -> Path:
        """Persist the analysis JSON to disk."""
        analysis = self.analyze()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
        return path

    def _aggregate_record_rl_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        aggregated: Dict[str, Dict[str, List[float]]] = {}
        for record in records:
            bucket = 'success' if record.get('success') else 'failure'
            for metric_name, raw_value in (record.get('rl_metrics') or {}).items():
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                metric_entry = aggregated.setdefault(metric_name, {'success': [], 'failure': []})
                metric_entry[bucket].append(value)

        cleaned: Dict[str, Dict[str, float]] = {}
        for metric_name, values in aggregated.items():
            cleaned_entry = {}
            for bucket_name, bucket_values in values.items():
                if bucket_values:
                    cleaned_entry[bucket_name] = sum(bucket_values) / len(bucket_values)
            if cleaned_entry:
                cleaned[metric_name] = cleaned_entry
        return cleaned

    def _aggregate_template_rl_metrics(self, raw_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        aggregated: Dict[str, Dict[str, float]] = {}
        for metric_name, buckets in raw_metrics.items():
            cleaned_entry = {}
            for bucket_name, bucket_values in buckets.items():
                if bucket_values:
                    cleaned_entry[bucket_name] = sum(bucket_values) / len(bucket_values)
            if cleaned_entry:
                aggregated[metric_name] = cleaned_entry
        return aggregated
