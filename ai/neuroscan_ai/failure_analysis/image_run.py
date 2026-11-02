"""Image review run for the aliases selected by the metrics-only stage (read-only inputs).

Order (no image is opened before step 4):
  1. approval phrase; 2. metrics-only result present and clean, alias count <= MAX_CASES,
  Final Test COMPLETED / not FAILED, output folder absent; 3. pre-access audit (time, HEAD,
  lock tag, Final Test snapshot hash, selection hash); 4. ``nibabel.load`` restricted to an
  allow-list of exactly the selected cases' DWI, ADC, GT and stored 2D / 2.5D predictions;
  5. per case: maps, components, matching, intensities, slices, figures, flags; 6. ID / path
  scan of public files, Final Test unchanged, staging renamed into place.
Nothing here runs a model, creates a probability map or writes into the Final Test folder.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neuroscan_ai.failure_analysis import audit as au
from neuroscan_ai.failure_analysis import image_review as ir
from neuroscan_ai.failure_analysis import schemas as sc

APPROVAL_PHRASE = "I APPROVE READ-ONLY FAILURE ANALYSIS OF SELECTED TEST CASES"
MAX_CASES = 10
LOCK_TAG = "neuroscan-v1.0.0-test-locked"
PROCESSED_ROOT = Path("data/processed/v1")
CATEGORIES = ("dwi", "adc", "gt", "prediction_2d", "prediction_25d")


class ImageReviewRefused(RuntimeError):
    pass


class NotAllowedFile(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8"
    )


def _default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def allow_list(allowed: dict[Path, str], counter: dict[str, int]) -> Iterator[None]:
    """Only the listed files can be opened with nibabel; every other attempt is refused."""
    import nibabel

    real = nibabel.load
    resolved = {p.resolve(): c for p, c in allowed.items()}

    def guarded(path, *a, **k):
        cat = resolved.get(Path(path).resolve())
        if cat is None:
            counter["refused"] = counter.get("refused", 0) + 1
            raise NotAllowedFile("file outside the approved selection")
        counter[cat] = counter.get(cat, 0) + 1
        return real(path, *a, **k)

    nibabel.load = guarded
    try:
        yield
    finally:
        nibabel.load = real


def preflight(fa_dir: Path, final_dir: Path, out: Path, phrase: str | None) -> dict[str, Any]:
    if phrase != APPROVAL_PHRASE:
        raise ImageReviewRefused("approval phrase missing or wrong; no image was opened")
    for name in ("selection.json", "audit.json", "_internal/aliases.json"):
        if not (fa_dir / name).is_file():
            raise ImageReviewRefused(f"metrics-only result incomplete ({name})")
    fa_audit = json.loads((fa_dir / "audit.json").read_text(encoding="utf-8"))
    if not fa_audit.get("public_output_original_id_scan", {}).get("passed"):
        raise ImageReviewRefused("metrics-only ID scan did not pass")
    aliases = json.loads((fa_dir / "_internal/aliases.json").read_text(encoding="utf-8"))
    selection = json.loads((fa_dir / "selection.json").read_text(encoding="utf-8"))
    if sorted(aliases) != sorted(a["alias"] for a in selection["aliases"]):
        raise ImageReviewRefused("alias map and selection disagree")
    if len(aliases) > MAX_CASES:
        raise ImageReviewRefused(f"{len(aliases)} aliases > maximum {MAX_CASES}")
    if not (final_dir / "COMPLETED").exists() or (final_dir / "FAILED").exists():
        raise ImageReviewRefused("final test result not completed")
    if out.exists():
        raise ImageReviewRefused(f"{out.name} exists; image review results are never overwritten")
    return {
        "aliases": aliases,
        "selection": selection,
        "selection_sha256": hashlib.sha256((fa_dir / "selection.json").read_bytes()).hexdigest(),
    }


def case_files(
    repo_root: Path, final_dir: Path, case_id: str, processed_root: Path = PROCESSED_ROOT
) -> dict[str, Path]:
    meta = json.loads(
        (repo_root / processed_root / case_id / "meta.json").read_text(encoding="utf-8")
    )
    src = meta["source"]
    return {
        "dwi": repo_root / src["dwi"],
        "adc": repo_root / src["adc"],
        "gt": repo_root / src["label"],
        "prediction_2d": final_dir / "predictions_2d" / f"{case_id}_pred.nii.gz",
        "prediction_25d": final_dir / "predictions_25d" / f"{case_id}_pred.nii.gz",
    }


def analyze_case(
    alias: str,
    files: dict[str, Path],
    stored: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    has_lesion: bool,
    case_dir: Path,
    figures: bool = True,
) -> dict[str, Any]:
    import nibabel as nib

    from neuroscan_ai.analysis.components import component_analysis, label_components
    from neuroscan_ai.preprocessing.mask import binarize
    from neuroscan_ai.research.compare import check_difference, difference_labelmap
    from neuroscan_ai.research.matching import match_components
    from neuroscan_ai.training.brain_gating import compute_brain_mask
    from neuroscan_ai.training.normalization import NormalizationSpec, normalize

    gt_img = nib.load(files["gt"])
    gt = binarize(gt_img.get_fdata()).astype(np.uint8)
    affine, spacing = gt_img.affine, [float(z) for z in gt_img.header.get_zooms()[:3]]
    imgs = {k: nib.load(files[k]) for k in ("dwi", "adc", "prediction_2d", "prediction_25d")}
    for k, img in imgs.items():
        if img.shape[:3] != gt.shape or not np.allclose(img.affine, affine, atol=1e-3):
            return {"alias": alias, "status": "stopped", "reason": f"{k} grid differs from GT"}
    dwi = imgs["dwi"].get_fdata(dtype=np.float32)
    adc = imgs["adc"].get_fdata(dtype=np.float32)
    preds = {
        "2d": (np.asanyarray(imgs["prediction_2d"].dataobj) > 0).astype(np.uint8),
        "25d": (np.asanyarray(imgs["prediction_25d"].dataobj) > 0).astype(np.uint8),
    }

    diffs, problems = {}, []
    for m in sc.MODELS:
        diffs[m] = difference_labelmap(preds[m], gt)
        issues = check_difference(diffs[m], preds[m], gt, stored[m])
        if issues:
            problems.append(f"{m}: {issues}")
    if problems:
        return {
            "alias": alias,
            "status": "stopped",
            "reason": "difference map counts differ from the stored Final Test metrics",
            "details": problems,
        }
    mdiff = ir.model_difference_map(preds["2d"], preds["25d"])

    brain = compute_brain_mask("skull_stripped_background", dwi)
    spec = NormalizationSpec("clip_zscore", (0.5, 99.5))
    norm = {"dwi": normalize(dwi, brain, spec), "adc": normalize(adc, brain, spec)}
    exclude = (gt > 0) | (preds["2d"] > 0) | (preds["25d"] > 0)

    maps = case_dir / "maps"
    maps.mkdir(parents=True)
    for name, arr in (
        ("difference_2d", diffs["2d"]),
        ("difference_25d", diffs["25d"]),
        ("model_difference", mdiff),
    ):
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), affine), str(maps / f"{name}.nii.gz"))

    gt_rows = ir.component_table(gt, spacing, affine, None, brain)
    comp, match, flags, intensity, boundary = {}, {}, {}, {}, {}
    for m in sc.MODELS:
        comp[m] = ir.component_table(preds[m], spacing, affine, gt, brain)
        match[m] = (
            match_components(preds[m], gt, spacing, affine, ir.SLICE_AXIS)
            if preds[m].any() or gt.any()
            else {"groups": []}
        )
        boundary[m] = ir.boundary_share(preds[m], gt, spacing)
        intensity[m] = ir.intensity_summary(norm, diffs[m], brain, exclude)
        flags[m] = ir.model_flags(stored[m], match[m], comp[m], gt_rows, boundary[m], has_lesion)
        flags[m]["T9"] = ir.t9_candidate(intensity[m])
    if gt.any():  # GT components: shortest distance (mm) to each model's prediction
        labels, _ = label_components(gt)
        _, gcomps = component_analysis(gt, spacing, affine, ir.SLICE_AXIS)
        by_rank = {c["size_rank"]: c["label"] for c in gcomps}
        for m in sc.MODELS:
            d = ir.distance_to(preds[m], spacing)
            for r in gt_rows:
                sel = labels == by_rank[r["component_id"]]
                r[f"nearest_{m}_mm"] = None if d is None else float(d[sel].min())
    for r in gt_rows:
        r.pop("nearest_other_mm", None)

    import pandas as pd

    pd.DataFrame(gt_rows).to_csv(case_dir / "components_gt.csv", index=False)
    for m in sc.MODELS:
        pd.DataFrame(comp[m]).to_csv(case_dir / f"components_{m}.csv", index=False)
        _json(case_dir / f"matching_{m}.json", match[m])
    _json(
        case_dir / "intensity.json",
        {
            "normalization": "case-wise clip_zscore over brain voxels (0.5 / 99.5 percentile), "
            "original grid; same function as training",
            "regions": "tp / fp / fn of each model; brain_reference = brain outside GT and both "
            "predictions",
            **{m: intensity[m] for m in sc.MODELS},
        },
    )

    reps = [
        r["representative_voxel"][ir.SLICE_AXIS] for r in (gt_rows or comp["2d"] or comp["25d"])[:2]
    ]
    slices = ir.select_slices(gt, diffs["2d"], diffs["25d"], mdiff, reps)
    n_fig = 0
    if figures:
        from neuroscan_ai.failure_analysis import figures as fg

        fdir = case_dir / "figures"
        fdir.mkdir()
        wins = {"dwi": fg.window(dwi, brain), "adc": fg.window(adc, brain)}
        for s in slices:
            fg.panel_a(
                fdir / f"{alias}_A_k{s['z']:03d}.png",
                alias,
                s["z"],
                dwi,
                adc,
                diffs["2d"],
                diffs["25d"],
                wins,
                spacing,
            )
            fg.panel_b(
                fdir / f"{alias}_B_k{s['z']:03d}.png",
                alias,
                s["z"],
                dwi,
                gt,
                mdiff,
                wins["dwi"],
                spacing,
            )
            n_fig += 2
        rows = gt_rows or comp["25d"] or comp["2d"]
        if rows:
            v = rows[0]["representative_voxel"]
            fg.panel_c(
                fdir / f"{alias}_C.png",
                alias,
                v,
                dwi,
                gt,
                diffs["2d"],
                diffs["25d"],
                wins["dwi"],
                spacing,
            )
            n_fig += 1
    return {
        "alias": alias,
        "status": "analyzed",
        "has_lesion": has_lesion,
        "metadata": meta,
        "spacing_mm": [round(s, 4) for s in spacing],
        "grid_shape": list(gt.shape),
        "components": {"gt": len(gt_rows), **{m: len(comp[m]) for m in sc.MODELS}},
        "matching_groups": {m: _types(match[m]) for m in sc.MODELS},
        "boundary": boundary,
        "flags": flags,
        "intensity": intensity,
        "slices": slices,
        "maps": 3,
        "figures": n_fig,
        "model_difference_voxels": {
            "both": int((mdiff == ir.MD_BOTH).sum()),
            "2d_only": int((mdiff == ir.MD_2D_ONLY).sum()),
            "25d_only": int((mdiff == ir.MD_25D_ONLY).sum()),
        },
    }


def _metrics_only_snapshot(fa_dir: Path, review_name: str) -> dict[str, str]:
    """Hashes of the metrics-only files, i.e. without the image-review folder and its staging."""
    return {
        k: v
        for k, v in au.snapshot(fa_dir).items()
        if not k.split("/")[0].lstrip(".").startswith(review_name)
    }


def _types(match: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in match.get("groups", []):
        out[g["group_type"]] = out.get(g["group_type"], 0) + 1
    return out


def run(
    fa_dir: Path,
    final_dir: Path,
    repo_root: Path,
    phrase: str | None,
    out: Path | None = None,
    figures: bool = True,
    processed_root: Path = PROCESSED_ROOT,
) -> dict[str, Any]:
    from neuroscan_ai.failure_analysis.review import summarize_case

    out = out or fa_dir / "image_review"
    pf = preflight(fa_dir, final_dir, out, phrase)
    torch_before = au.torch_loaded()
    ft_before = au.snapshot(final_dir)
    fa_before = _metrics_only_snapshot(fa_dir, out.name)
    pre = {
        "approved_at": _now(),
        "approval_phrase_confirmed": True,
        "git_head": au.git(repo_root, "rev-parse", "HEAD"),
        "lock_tag": LOCK_TAG,
        "lock_tag_commit": au.git(repo_root, "rev-parse", f"refs/tags/{LOCK_TAG}^{{commit}}"),
        "final_test_snapshot_sha256": _sha(ft_before),
        "metrics_only_selection_sha256": pf["selection_sha256"],
        "selected_aliases": len(pf["aliases"]),
        "max_cases": MAX_CASES,
    }
    staging = out.parent / f".{out.name}.staging-{os.getpid()}"
    if staging.exists():
        raise ImageReviewRefused("staging folder exists")
    counter: dict[str, int] = {}
    try:
        staging.mkdir(parents=True)
        allowed: dict[Path, str] = {}
        files_by_alias = {}
        for alias, cid in sorted(pf["aliases"].items()):
            files = case_files(repo_root, final_dir, cid, processed_root)
            files_by_alias[alias] = files
            allowed.update({p: c for c, p in files.items()})
        results, summaries = [], []
        with allow_list(allowed, counter):
            for alias in sorted(pf["aliases"]):
                s = json.loads(
                    (fa_dir / "cases" / alias / "summary.json").read_text(encoding="utf-8")
                )
                stored = s["observations"]["metrics"]
                res = analyze_case(
                    alias,
                    files_by_alias[alias],
                    stored,
                    s["observations"]["metadata"],
                    s["has_lesion"],
                    staging / "cases" / alias,
                    figures,
                )
                res["groups"] = s["groups"]
                res["dice"] = {m: stored[m]["dice"] for m in sc.MODELS}
                res["dice_difference"] = s["observations"]["dice_difference_25d_minus_2d"]
                res["t8_metrics_only"] = s["taxonomy_flags"]["T8"]
                results.append(res)
                summary = summarize_case(res)
                summaries.append(summary)
                _json(staging / "cases" / alias / "summary.json", summary)
        from neuroscan_ai.failure_analysis.review import aggregate

        _json(staging / "aggregate_observations.json", aggregate(results))
        _json(
            staging / "_internal" / "case_access.json",
            {
                "aliases": {a: pf["aliases"][a] for a in sorted(pf["aliases"])},
                "files_opened_per_category": {c: counter.get(c, 0) for c in CATEGORIES},
                "refused_attempts": counter.get("refused", 0),
            },
        )
        ids = list(pf["aliases"].values())
        root_str = str(repo_root.resolve())
        hits = au.find_ids(staging, ids)
        paths = [
            p.relative_to(staging).as_posix()
            for p in staging.rglob("*")
            if p.is_file()
            and "_internal" not in p.parts
            and p.suffix in (".json", ".csv")
            and (
                root_str in p.read_text(encoding="utf-8")
                or root_str.replace("\\", "/") in p.read_text(encoding="utf-8")
            )
        ]
        named = [p.name for p in staging.rglob("*") if any(i in p.name for i in ids)]
        ft_after = au.snapshot(final_dir)
        fa_after = _metrics_only_snapshot(fa_dir, out.name)
        audit = {
            "mode": "image-review",
            **pre,
            "finished_at": _now(),
            "aliases_accessed": len(results),
            "aliases_analyzed": sum(r["status"] == "analyzed" for r in results),
            "aliases_stopped": [
                {"alias": r["alias"], "reason": r["reason"]}
                for r in results
                if r["status"] != "analyzed"
            ],
            "files_opened": {c: counter.get(c, 0) for c in CATEGORIES},
            "refused_file_attempts": counter.get("refused", 0),
            "model_inference_runs": 0,
            "probability_maps_generated": 0,
            "torch_imported_by_this_run": (not torch_before) and au.torch_loaded(),
            "difference_maps_written": sum(r.get("maps", 0) for r in results),
            "figures_written": sum(r.get("figures", 0) for r in results),
            "final_test_unchanged": ft_before == ft_after,
            "metrics_only_files_unchanged": fa_before == fa_after,
            "public_output_original_id_scan": {
                "files_with_ids": hits,
                "file_names_with_ids": named,
                "files_with_local_paths": paths,
                "passed": not (hits or named or paths),
            },
            "output_git_ignored": au.git_ignored(repo_root, out),
        }
        _json(staging / "audit.json", audit)
        if (
            hits
            or named
            or paths
            or not audit["final_test_unchanged"]
            or not audit["metrics_only_files_unchanged"]
            or audit["refused_file_attempts"]
        ):
            raise ImageReviewRefused(
                f"post-run checks failed: {audit['public_output_original_id_scan']}"
            )
        staging.rename(out)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"out": out, "audit": audit, "results": results, "summaries": summaries}
