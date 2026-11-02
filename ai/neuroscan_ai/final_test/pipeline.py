"""Final Test orchestration: pre-flight -> approval -> run state -> evaluation -> reports.

Order (nothing of the evaluated split is opened before step 5):
  1. manifest + hashes (checkpoints, configs, split) + checkpoint info + frozen policy + Git
  2. final-test only: --approve-test == manifest SHA-256, confirmation phrase, fixed output dir
  3. output directory state (LOCK / COMPLETED / FAILED / results) and the resume rules
  4. atomic LOCK
  5. SplitGuard(allow_test=True) for the final test only (dry-run: validation, guard sealed),
     the training-validation evaluation path for both models, reports into a staging folder
  6. COMPLETED (staging moved into place) or FAILED (staging moved to failed_attempts/attempt-<n>)

Console output never contains case IDs; in final-test mode it contains no metric either.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuroscan_ai.final_test import analysis
from neuroscan_ai.final_test import manifest as mf
from neuroscan_ai.final_test import run_state as rs

MODES = ("validation-dry-run", "final-test")
SPLIT_FOR_MODE = {"validation-dry-run": "val", "final-test": "test"}


class PreflightError(RuntimeError):
    def __init__(self, message: str, checks: list[mf.Check] | None = None) -> None:
        super().__init__(message)
        self.checks = checks or []


@dataclass
class RunOptions:
    mode: str = "validation-dry-run"
    repo_root: Path = Path(".")
    manifest: Path = mf.MANIFEST_PATH
    output_dir: Path | None = None
    approve_test: str | None = None
    confirm_phrase: str | None = None
    resume_after_failure: bool = False
    failure_report: Path | None = None
    reason: str | None = None
    device: str = "auto"
    qa_images: bool = True
    command: list[str] = field(default_factory=list)
    approved_at: str | None = None


# --- 1-3 pre-flight -----------------------------------------------------------------------------


def preflight(opts: RunOptions) -> dict[str, Any]:
    if opts.mode not in MODES:
        raise PreflightError(f"unknown mode {opts.mode}")
    root = opts.repo_root.resolve()
    manifest_path = root / opts.manifest
    manifest = mf.load_manifest(manifest_path)
    manifest_sha = mf.sha256_text(manifest_path)
    cfgs = mf.load_model_configs(manifest, root)

    checks = mf.check_artifacts(manifest, root)
    checks += mf.check_checkpoint_info(manifest, root)
    checks += mf.check_policy(manifest, cfgs)
    checks += mf.check_git(manifest, root, opts.manifest.as_posix())

    test_out = root / manifest["rules"]["output_dir"]
    if opts.mode == "final-test":
        checks.append(
            mf.Check(
                "approve_test_matches_manifest_sha256",
                opts.approve_test == manifest_sha,
                "ok"
                if opts.approve_test == manifest_sha
                else "missing or different --approve-test value",
            )
        )
        checks.append(
            mf.Check(
                "confirmation_phrase",
                opts.confirm_phrase == mf.CONFIRM_PHRASE,
                "ok"
                if opts.confirm_phrase == mf.CONFIRM_PHRASE
                else "confirmation phrase missing or wrong",
            )
        )
        fixed = opts.output_dir is None or opts.output_dir.resolve() == test_out.resolve()
        checks.append(
            mf.Check("output_dir_is_manifest_output_dir", fixed, manifest["rules"]["output_dir"])
        )
        out = test_out
    else:
        if opts.approve_test or opts.confirm_phrase:
            checks.append(
                mf.Check(
                    "no_test_approval_in_dry_run",
                    False,
                    "approval arguments are only accepted with --mode final-test",
                )
            )
        out = (opts.output_dir or (root / manifest["rules"]["dry_run_output_dir"])).resolve()
        inside = out == test_out.resolve() or test_out.resolve() in out.parents
        checks.append(mf.Check("dry_run_output_outside_test_dir", not inside, out.as_posix()))

    blocking = mf.blocking_failures(checks, opts.mode)
    if blocking:
        raise PreflightError("pre-flight refused: " + ", ".join(c.name for c in blocking), checks)
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "cfgs": cfgs,
        "checks": checks,
        "out": out,
    }


# --- environment --------------------------------------------------------------------------------


def environment(device_type: str) -> dict[str, Any]:
    import monai
    import nibabel
    import scipy
    import torch

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "monai": monai.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "nibabel": nibabel.__version__,
        "pandas": pd.__version__,
        "device": device_type,
    }
    if device_type == "cuda":
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "name": p.name,
            "total_vram_gb": round(p.total_memory / 1024**3, 2),
            "capability": f"{p.major}.{p.minor}",
        }
    return env


# --- 5 evaluation -------------------------------------------------------------------------------


def _device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _subgroup_meta(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = root / manifest["subgroups"]["metadata"]
    if not path.is_file():
        return {}
    md = pd.read_csv(path).set_index("case_id")
    cols = [c for c in manifest["subgroups"]["columns"] if c in md.columns]
    return {cid: {c: row[c] for c in cols} for cid, row in md[cols].iterrows()}


def evaluate_models(
    pf: dict[str, Any],
    split: str,
    allow_test: bool,
    device_name: str,
    staging: Path,
    audit: dict[str, Any],
) -> dict[str, Any]:
    import nibabel as nib
    import torch

    from neuroscan_ai.data.split_guard import SplitGuard
    from neuroscan_ai.training.dataset import load_split_cases, norm_specs
    from neuroscan_ai.training.inference import load_original_gt
    from neuroscan_ai.training.seed import seed_everything
    from neuroscan_ai.training.trainer import load_model_from_checkpoint, validate

    root, manifest, cfgs = pf["root"], pf["manifest"], pf["cfgs"]
    seed_info = seed_everything(manifest["policy"]["seed"], True)
    device = _device(device_name)
    guard = SplitGuard(root / manifest["split"]["path"], allow_test=allow_test)
    data = cfgs["2d"]["data"]
    t0 = time.time()
    cases = load_split_cases(
        guard,
        split,
        root / data["processed_root"],
        data["modalities"],
        norm_specs(data["normalization"]),
        None,
        data["brain_mask_policy"],
    )
    gts = {c.case_id: load_original_gt(c.meta, root) for c in cases}
    audit["cases_accessed"] = len(cases)
    audit["load_time_s"] = round(time.time() - t0, 1)
    meta = _subgroup_meta(root, manifest)

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    preds_by_model: dict[str, dict[str, np.ndarray]] = {}
    audit["models"] = {}
    for key in mf.MODEL_KEYS:
        cfg, m = cfgs[key], manifest["models"][key]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        model, info = load_model_from_checkpoint(root / m["checkpoint"], cfg, device)
        if info.get("config_hash") != m["config_hash"]:
            raise RuntimeError(f"{key}: checkpoint config hash changed after pre-flight")
        result = validate(model, cases, gts, cfg, device, None, keep_predictions=True)
        rows = result.rows
        for r in rows:
            r.update({c: v for c, v in meta.get(r["case_id"], {}).items()})
        rows_by_model[key] = rows
        preds_by_model[key] = result.predictions
        pred_dir = staging / f"predictions_{key}"
        pred_dir.mkdir()
        for cid, arr in result.predictions.items():
            nib.save(
                nib.Nifti1Image(arr.astype(np.uint8), gts[cid][1].affine),
                str(pred_dir / f"{cid}_pred.nii.gz"),
            )
        audit["models"][key] = {
            "checkpoint_sha256": m["checkpoint_sha256"],
            "time_s": round(time.time() - t, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            if device.type == "cuda"
            else 0.0,
            "cases_succeeded": len(rows),
        }
        del model
    report = {**guard.access_report(), "split_file": manifest["split"]["path"]}
    audit["guard_access"] = {k: v for k, v in report.items() if k != "test_cases_accessed"}
    audit["test_cases_accessed_count"] = len(report["test_cases_accessed"])
    internal = {
        "split": split,
        "case_ids": sorted(c.case_id for c in cases),
        "guard_access_report": report,
    }
    return {
        "rows": rows_by_model,
        "preds": preds_by_model,
        "cases": cases,
        "gts": gts,
        "device": device,
        "seed": seed_info,
        "internal_access": internal,
        "meta": meta,
    }


def write_reports(pf: dict[str, Any], ev: dict[str, Any], staging: Path) -> dict[str, Any]:
    manifest = pf["manifest"]
    rows = ev["rows"]
    st = manifest["statistics"]["bootstrap"]
    sub_cols = [c for c in manifest["subgroups"]["columns"]]
    for key in mf.MODEL_KEYS:
        pd.DataFrame(rows[key]).to_csv(staging / f"patient_metrics_{key}.csv", index=False)
    table = analysis.paired_positive_table(rows["2d"], rows["25d"], ev["meta"], sub_cols)
    table.to_csv(staging / "paired_metrics.csv", index=False)
    agg = analysis.aggregate({"2d": rows["2d"], "25d": rows["25d"]})
    stats = analysis.statistical_comparison(table, st["n_boot"], st["seed"])
    subs = analysis.subgroup_summary(
        table,
        [c for c in sub_cols if c in table.columns],
        manifest["subgroups"]["min_n_for_inference"],
        st["n_boot"],
        st["seed"],
    )
    neg = analysis.negative_summary(rows["2d"], rows["25d"])
    thr = float(manifest["failure_case"]["below"])
    fails = analysis.failure_cases(table, thr)
    qa = analysis.qa_selection(table, neg, 3, thr)
    _json(staging / "aggregate_metrics.json", agg)
    _json(staging / "statistical_comparison.json", stats)
    _json(staging / "subgroup_metrics.json", subs)
    _json(staging / "negative_case_summary.json", neg)
    _json(
        staging / "qa_manifest.json",
        {
            "rules": manifest["qa_images"],
            "failure_rule": manifest["failure_case"],
            "failure_cases": fails,
            "selected": qa,
        },
    )
    return {
        "table": table,
        "aggregate": agg,
        "stats": stats,
        "negative": neg,
        "qa": qa,
        "failures": fails,
    }


def qa_figures(ev: dict[str, Any], qa: list[dict[str, Any]], root: Path, outdir: Path) -> int:
    """One PNG per selected case: original-grid DWI, GT / 2D / 2.5D outlines on the slice with
    the largest GT (or prediction, for negatives) area."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    outdir.mkdir()
    cases = {c.case_id: c for c in ev["cases"]}
    for i, item in enumerate(qa, 1):
        cid = item["case_id"]
        gt, _ = ev["gts"][cid]
        p2, p25 = ev["preds"]["2d"][cid], ev["preds"]["25d"][cid]
        ref = gt if gt.any() else (p2 | p25 if (p2.any() or p25.any()) else gt)
        z = int(np.argmax(ref.reshape(-1, ref.shape[2]).sum(axis=0)))
        dwi = nib.load(root / cases[cid].meta["source"]["dwi"]).get_fdata()[:, :, z]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(dwi.T, cmap="gray", origin="lower")
        for arr, color in ((gt, "magenta"), (p2, "deepskyblue"), (p25, "orange")):
            if arr[:, :, z].any():
                ax.contour(arr[:, :, z].T, levels=[0.5], colors=color, linewidths=0.8)
        ax.set_title(f"QA {i:02d} z={z}  GT magenta / 2D blue / 2.5D orange", fontsize=7)
        ax.axis("off")
        fig.savefig(outdir / f"qa_{i:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        item["image"] = f"qa_images/qa_{i:02d}.png"
        item["slice_z"] = z
    return len(qa)


def reference_match(
    pf: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    stats: dict[str, Any],
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Dry-run only: compare with the stored Phase 3 / Phase 4 validation results."""
    root, manifest = pf["root"], pf["manifest"]
    out: dict[str, Any] = {"tolerance": tol, "models": {}}
    ok_all = True
    cols = (
        "dice",
        "iou",
        "sensitivity",
        "precision",
        "volume_error_ml",
        "component_count_prediction",
        "prediction_voxels",
        "tp_voxels",
        "fp_voxels",
    )
    for key in mf.MODEL_KEYS:
        ref_path = root / manifest["models"][key]["validation_reference"] / "validation_metrics.csv"
        if not ref_path.is_file():
            out["models"][key] = {"status": "reference missing"}
            ok_all = False
            continue
        ref = pd.read_csv(ref_path).set_index("case_id")
        new = pd.DataFrame(rows[key]).set_index("case_id")
        same_cases = set(ref.index) == set(new.index)
        diffs: dict[str, float] = {}
        for c in cols if same_cases else ():
            a = pd.to_numeric(new.loc[ref.index, c], errors="coerce").to_numpy(float)
            b = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            d = np.where(both_nan, 0.0, np.abs(a - b))
            diffs[c] = float(np.nanmax(np.where(np.isnan(d), np.inf, d)))
        ok = same_cases and all(v <= tol for v in diffs.values())
        ok_all &= ok
        out["models"][key] = {
            "same_cases": same_cases,
            "n": int(len(ref)),
            "max_abs_difference": diffs,
            "match": ok,
        }
    names = {k: Path(manifest["models"][k]["validation_reference"]).name for k in mf.MODEL_KEYS}
    comp = root / "results/phase4/comparison" / f"{names['25d']}_vs_{names['2d']}"
    comps = [comp / "summary.json"] if (comp / "summary.json").is_file() else []
    if comps:
        ref_sum = json.loads(comps[0].read_text(encoding="utf-8"))["positive_overall"]["dice"]
        new = stats["primary"]
        fields = ("mean_base", "mean_cand", "mean_delta", "median_delta", "wilcoxon_p")
        d = {f: abs(float(new[f]) - float(ref_sum[f])) for f in fields}
        d["ci95_low"] = abs(new["ci95"][0] - ref_sum["ci95"][0])
        d["ci95_high"] = abs(new["ci95"][1] - ref_sum["ci95"][1])
        cmp_ok = all(v <= tol for v in d.values())
        ok_all &= cmp_ok
        out["phase4_comparison"] = {
            "reference": comps[0].parent.name,
            "max_abs_difference": d,
            "match": cmp_ok,
        }
    else:
        out["phase4_comparison"] = {"status": "reference missing"}
        ok_all = False
    out["match"] = ok_all
    return out


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# --- 4-6 run ------------------------------------------------------------------------------------


def run(opts: RunOptions, evaluate=evaluate_models) -> dict[str, Any]:
    pf = preflight(opts)
    out: Path = pf["out"]
    attempt = rs.check_output_dir(out, opts.resume_after_failure)
    resume_info = None
    if opts.resume_after_failure:
        dest = rs.archive_failed_attempt(out, opts.failure_report or Path(""), opts.reason or "")
        resume_info = {"previous_attempt": dest.relative_to(out).as_posix(), "reason": opts.reason}
    head = mf.head_commit(pf["root"])
    audit: dict[str, Any] = {
        "mode": opts.mode,
        "attempt": attempt,
        "resume": resume_info,
        "approved_at": opts.approved_at,
        "command": opts.command,
        "git_commit": head,
        "lock_tag": pf["manifest"]["git"].get("lock_tag"),
        "frozen_commit": pf["manifest"]["git"].get("frozen_commit"),
        "manifest_sha256": pf["manifest_sha256"],
        "checkpoint_sha256": {
            k: pf["manifest"]["models"][k]["checkpoint_sha256"] for k in mf.MODEL_KEYS
        },
        "config_hash": {k: pf["manifest"]["models"][k]["config_hash"] for k in mf.MODEL_KEYS},
        "split_sha256": pf["manifest"]["split"]["sha256"],
        "preflight_checks": [c.as_dict() for c in pf["checks"]],
    }
    rs.acquire_lock(
        out,
        {
            "mode": opts.mode,
            "attempt": attempt,
            "manifest_sha256": pf["manifest_sha256"],
            "git_commit": head,
        },
    )
    staging = rs.staging_dir(out, attempt)
    audit["started_at"] = rs.now()
    split = SPLIT_FOR_MODE[opts.mode]
    try:
        ev = evaluate(pf, split, opts.mode == "final-test", opts.device, staging, audit)
        rep = write_reports(pf, ev, staging)
        if opts.qa_images:
            qa_figures(ev, rep["qa"], pf["root"], staging / "qa_images")
            _json(
                staging / "qa_manifest.json",
                {
                    "rules": pf["manifest"]["qa_images"],
                    "failure_rule": pf["manifest"]["failure_case"],
                    "failure_cases": rep["failures"],
                    "selected": rep["qa"],
                },
            )
        match = None
        if opts.mode == "validation-dry-run":
            match = reference_match(pf, ev["rows"], rep["stats"])
            _json(staging / "reference_match.json", match)
        env = environment(ev["device"].type)
        env["seed"] = ev["seed"]
        _json(staging / "environment.json", env)
        internal = staging / "audit_internal"
        internal.mkdir()
        _json(internal / "case_access.json", ev["internal_access"])
        audit.update(
            {
                "finished_at": rs.now(),
                "status": "completed",
                "cases_succeeded": min(m["cases_succeeded"] for m in audit["models"].values()),
                "cases_failed": 0,
            }
        )
        _json(staging / "execution_audit.json", audit)
        rs.mark_completed(
            out, staging, {"attempt": attempt, "manifest_sha256": pf["manifest_sha256"]}
        )
    except BaseException as exc:  # noqa: BLE001 - every failure must leave a FAILED record
        reason = f"{type(exc).__name__}: {exc}"
        audit.update({"finished_at": rs.now(), "status": "failed", "abort_reason": reason})
        if staging.exists():
            _json(staging / "execution_audit.json", audit)
            (staging / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        rs.mark_failed(out, staging, attempt, reason)
        raise
    return {"out": out, "attempt": attempt, "report": rep, "match": match, "audit": audit}
