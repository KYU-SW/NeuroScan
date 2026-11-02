"""Metrics-only run: completed Final Test tables -> selection, aliases, flags, summaries, audit.

Output (results/failure_analysis/v1/, Git-ignored); written into a staging folder that is renamed
into place only after the public files passed the original-ID scan:

  _internal/aliases.json, _internal/case_access.json   original IDs (local only)
  selection.json, overview.csv, aggregate_summary.json, audit.json, cases/<alias>/summary.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neuroscan_ai.failure_analysis import audit as au
from neuroscan_ai.failure_analysis import schemas as sc
from neuroscan_ai.failure_analysis import selection as sel
from neuroscan_ai.failure_analysis import taxonomy as tx

LOCK_TAG = "neuroscan-v1.0.0-test-locked"


class FailureAnalysisError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def check_final_test(final_dir: Path) -> None:
    if not final_dir.is_dir():
        raise FailureAnalysisError("final test result folder not found")
    bad = [n for n in sc.FORBIDDEN_FILES if (final_dir / n).exists()]
    missing = [n for n in sc.REQUIRED_FILES if not (final_dir / n).exists()]
    if bad or missing:
        raise FailureAnalysisError(
            f"final test result incomplete: missing {missing}, unexpected {bad}"
        )


def _records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import math

    out = []
    for r in rows:
        out.append(
            {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        )
    return out


def run(final_dir: Path, out_dir: Path, repo_root: Path) -> dict[str, Any]:
    import pandas as pd

    check_final_test(final_dir)
    if out_dir.exists():
        raise FailureAnalysisError(
            f"{out_dir.as_posix()} exists; metrics-only results are never overwritten"
        )
    started = _now()
    torch_before = au.torch_loaded()
    counter: dict[str, int] = {"nifti_open_attempts": 0}
    before = au.snapshot(final_dir)
    staging = out_dir.parent / f".{out_dir.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FailureAnalysisError(f"{staging.as_posix()} exists")
    try:
        with au.no_image_access(counter):
            rows = {
                m: _records(pd.read_csv(final_dir / f"patient_metrics_{m}.csv").to_dict("records"))
                for m in sc.MODELS
            }
            qa = json.loads((final_dir / "qa_manifest.json").read_text(encoding="utf-8"))
            records = sel.build_records(rows["2d"], rows["25d"])
            groups = sel.select(records)
            ordered = sel.assign(records, groups)
            summaries = [tx.summarize(r) for r in ordered]
            alias_of = {r.case_id: r.alias for r in ordered}

            staging.mkdir(parents=True)
            _json(staging / "_internal" / "aliases.json", {r.alias: r.case_id for r in ordered})
            _json(
                staging / "_internal" / "case_access.json",
                {
                    "source": "final test patient metrics tables (CSV only)",
                    "case_ids_read_from_tables": sorted(records),
                    "selected_case_ids": sorted(alias_of),
                },
            )
            _json(
                staging / "selection.json",
                {
                    "mode": sc.MODE_METRICS_ONLY,
                    "group_order": list(sc.GROUP_ORDER),
                    "rules": sc.GROUP_RULES,
                    "groups": {g: [alias_of[c] for c in groups.get(g, [])] for g in sc.GROUP_ORDER},
                    "aliases": [
                        {"alias": r.alias, "groups": r.groups, "reasons": r.reasons}
                        for r in ordered
                    ],
                    "tie_break": "metric order, then |difference|, then internal case order",
                },
            )
            pd.DataFrame(tx.overview_rows(summaries)).to_csv(staging / "overview.csv", index=False)
            agg = tx.aggregate(summaries, groups)
            qa_fail = {f["case_id"] for f in qa.get("failure_cases", [])}
            agg["consistent_with_final_test_failure_cases"] = qa_fail == set(groups["F"])
            _json(staging / "aggregate_summary.json", agg)
            for s in summaries:
                _json(staging / "cases" / s["alias"] / "summary.json", s)

            hits = au.find_ids(staging, records)
            if hits:
                raise FailureAnalysisError(f"original case IDs in public files: {hits}")
        after = au.snapshot(final_dir)
        unchanged = before == after
        if not unchanged:
            raise FailureAnalysisError("final test result folder changed during the run")
        extra = [
            p.relative_to(staging).as_posix()
            for p in staging.rglob("*")
            if p.is_file() and p.suffix.lower() in (".nii", ".gz", ".png", ".pt")
        ]
        commit = au.git(repo_root, "rev-parse", "HEAD")
        tag_commit = au.git(repo_root, "rev-parse", f"refs/tags/{LOCK_TAG}^{{commit}}")
        audit = {
            "mode": sc.MODE_METRICS_ONLY,
            "started_at": started,
            "finished_at": _now(),
            "git_commit": commit,
            "final_test_lock_tag": LOCK_TAG,
            "final_test_lock_tag_commit": tag_commit,
            "final_test_result": {
                "completed": True,
                "failed_marker": False,
                "files_hashed": len(before),
                "unchanged_after_run": unchanged,
            },
            "unique_aliases": len(ordered),
            "group_counts": agg["group_counts"],
            "access": {
                "test_image_access": 0,
                "dwi_access": 0,
                "adc_access": 0,
                "gt_nifti_access": 0,
                "prediction_nifti_access": 0,
                "nifti_open_attempts_refused": counter["nifti_open_attempts"],
            },
            "model_inference_runs": 0,
            "probability_maps_generated": 0,
            "torch_imported_by_this_run": (not torch_before) and au.torch_loaded(),
            "image_or_map_files_written": extra,
            "public_output_original_id_scan": {"files_with_ids": hits, "passed": not hits},
            "output_git_ignored": au.git_ignored(repo_root, out_dir),
            "command": ["scripts/analyze_failure_cases.py", *sys.argv[1:]],
        }
        _json(staging / "audit.json", audit)
        hits = au.find_ids(staging, records)
        if hits or counter["nifti_open_attempts"] or extra:
            raise FailureAnalysisError("post-run checks failed")
        staging.rename(out_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"out": out_dir, "aggregate": agg, "audit": audit, "summaries": summaries}
