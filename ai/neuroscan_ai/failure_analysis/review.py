"""Image-review case summaries and aggregate (observations separated from hypothesis templates).

``observations`` hold measured values with IDs (O-..). ``taxonomy_flags`` and
``review_candidates`` are rule outputs. ``hypotheses`` only contain *templates* triggered by a
flag, each with evidence observation IDs, alternative explanations, a verification method and a
confidence of ``possible`` or ``undetermined`` — never a confirmed cause, diagnosis or a
statement that the ground truth is wrong.
"""

from __future__ import annotations

from typing import Any

from neuroscan_ai.failure_analysis import schemas as sc

MODEL_NAMES = {"2d": "2D", "25d": "2.5D"}


def _obs(out: list, oid: str, measure: str, value: Any, model: str | None = None) -> str:
    out.append(
        {"id": oid, "measure": measure, "model": model, "value": value, "status": "observed"}
    )
    return oid


def summarize_case(res: dict[str, Any]) -> dict[str, Any]:
    base = {"alias": res["alias"], "groups": res.get("groups", []), "status": res["status"]}
    if res["status"] != "analyzed":
        return {
            **base,
            "reason": res.get("reason"),
            "observations": [],
            "taxonomy_flags": {},
            "review_candidates": [],
            "hypotheses": [],
            "alternative_explanations": [],
            "unknowns": ["case not analysed: " + str(res.get("reason"))],
            "suggested_experiments": [],
        }
    o: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    ids["meta"] = _obs(o, "O-01", "metadata_groups", res["metadata"])
    ids["dice"] = _obs(
        o,
        "O-02",
        "stored_final_test_dice",
        {**res["dice"], "difference_25d_minus_2d": res["dice_difference"]},
    )
    ids["comp"] = _obs(o, "O-03", "component_count", res["components"])
    ids["mdiff"] = _obs(o, "O-04", "model_difference_voxels", res["model_difference_voxels"])
    n = 5
    for m in sc.MODELS:
        f = res["flags"][m]
        ids[f"match_{m}"] = _obs(
            o, f"O-{n:02d}", "matching_group_types", res["matching_groups"][m], m
        )
        ids[f"bound_{m}"] = _obs(
            o,
            f"O-{n + 1:02d}",
            "fp_fn_share_within_2mm_of_gt_boundary",
            res["boundary"][m]["share"],
            m,
        )
        ids[f"small_{m}"] = _obs(
            o,
            f"O-{n + 2:02d}",
            "small_gt_components_missed_lt_0.1ml",
            f.get("small_gt_components_missed"),
            m,
        )
        ids[f"far_{m}"] = _obs(
            o,
            f"O-{n + 3:02d}",
            "prediction_only_components_gt_10mm_from_gt",
            f.get("distant_fp_components"),
            m,
        )
        med = {
            mod: {
                r: res["intensity"][m][mod][r]["median"]
                for r in ("tp", "fp", "fn", "brain_reference")
            }
            for mod in ("dwi", "adc")
        }
        ids[f"int_{m}"] = _obs(o, f"O-{n + 4:02d}", "median_zscore_by_region", med, m)
        n += 5
    ids["slices"] = _obs(
        o,
        f"O-{n:02d}",
        "review_centre_slices",
        [{"k": s["z"], "reasons": s["reasons"]} for s in res["slices"]],
    )

    flags = {
        m: {c: res["flags"][m][c] for c in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")}
        | {"T9": res["flags"][m]["T9"]["flag"]}
        for m in sc.MODELS
    }
    flags["T8"] = res["t8_metrics_only"]
    cands = []
    for m in sc.MODELS:
        if flags[m]["T6"] == sc.REVIEW_CANDIDATE:
            cands.append({"code": "T6", "model": m, "evidence": [ids[f"bound_{m}"]]})
        if flags[m]["T9"] == sc.REVIEW_CANDIDATE:
            cands.append(
                {
                    "code": "T9",
                    "model": m,
                    "reasons": res["flags"][m]["T9"]["reasons"],
                    "evidence": [ids[f"int_{m}"]],
                }
            )
    if flags["T8"] == sc.REVIEW_CANDIDATE:
        cands.append(
            {"code": "T8", "model": None, "evidence": [ids["meta"], ids["dice"], ids["mdiff"]]}
        )
    return {
        **base,
        "has_lesion": res["has_lesion"],
        "observations": o,
        "taxonomy_flags": flags,
        "review_candidates": cands,
        "hypotheses": hypothesis_templates(flags, ids),
        "alternative_explanations": [
            "single-seed training variation",
            "label boundary variability",
            "original-space restoration loss of 1-2 voxel components (Phase 2 limitation)",
        ],
        "unknowns": [
            "clinical meaning of any signal pattern (not assessed)",
            "whether the ground truth extent is complete (not assessed)",
            "model confidence near the threshold (no probability map)",
        ],
        "suggested_experiments": [],
        "note": "Observations are measurements; templates are review prompts, not findings.",
    }


def hypothesis_templates(flags: dict[str, Any], ids: dict[str, str]) -> list[dict[str, Any]]:
    out = []

    def add(hid, text, evidence, alternatives, verify, confidence="undetermined"):
        out.append(
            {
                "id": hid,
                "status": "template",
                "statement": text,
                "evidence": evidence,
                "alternative_explanations": alternatives,
                "verification": verify,
                "confidence": confidence,
            }
        )

    if flags["T8"] == sc.REVIEW_CANDIDATE:
        add(
            "H-T8",
            "With thick slices the adjacent-slice input of the 2.5D model may have "
            "added less useful context than for thin slices.",
            [ids["meta"], ids["dice"], ids["mdiff"]],
            [
                "single-seed variation",
                "channel-wise independent intensity augmentation (R1)",
                "case-specific label boundary differences",
            ],
            [
                "multi-seed training",
                "thick-slice cases in external data",
                "modality-consistent augmentation ablation",
            ],
        )
    for m in sc.MODELS:
        if flags[m]["T4"] == sc.FLAGGED:
            add(
                f"H-T4-{m}",
                f"{MODEL_NAMES[m]} may have a detection limit for GT components "
                "below 0.1 mL in this case.",
                [ids[f"small_{m}"], ids["comp"]],
                [
                    "1-2 voxel component loss in resampling / restoration",
                    "label boundary variability for very small regions",
                ],
                ["small-component audit across validation cases", "external small-lesion data"],
                "possible",
            )
        if flags[m]["T5"] == sc.FLAGGED:
            add(
                f"H-T5-{m}",
                f"{MODEL_NAMES[m]} marked regions more than 10 mm from any GT "
                "component; their signal pattern needs review.",
                [ids[f"far_{m}"], ids[f"int_{m}"]],
                ["signal near brain boundary or artefacts", "signal change outside the GT extent"],
                ["image review of the marked regions", "check against brain-boundary distance"],
            )
        if flags[m]["T9"] == sc.REVIEW_CANDIDATE:
            add(
                f"H-T9-{m}",
                f"The relation between DWI / ADC signal and the GT extent in the "
                f"{MODEL_NAMES[m]} FP or FN regions needs a closer look.",
                [ids[f"int_{m}"]],
                ["normalisation range of this case", "surrounding tissue signal"],
                ["independent visual review", "comparison with neighbouring slices"],
            )
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r["status"] == "analyzed"]
    codes = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

    def flag_of(r, m, c):
        return r["flags"][m]["T9"]["flag"] if c == "T9" else r["flags"][m][c]

    per_model = {
        m: {
            c: {
                v: sum(flag_of(r, m, c) == v for r in ok)
                for v in (sc.FLAGGED, sc.REVIEW_CANDIDATE, sc.NOT_FLAGGED, sc.NOT_APPLICABLE)
            }
            for c in (*codes, "T9")
        }
        for m in sc.MODELS
    }
    match_types = {m: {} for m in sc.MODELS}
    for r in ok:
        for m in sc.MODELS:
            for t, n in r["matching_groups"][m].items():
                match_types[m][t] = match_types[m].get(t, 0) + n

    def by(key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for r in ok:
            k = str(r["metadata"].get(key))
            e = out.setdefault(k, {"n": 0, "aliases": []})
            e["n"] += 1
            e["aliases"].append(r["alias"])
        return out

    def group_pattern(g: str) -> dict[str, Any]:
        rs = [r for r in ok if g in r.get("groups", [])]
        return {
            "n": len(rs),
            "aliases": [r["alias"] for r in rs],
            "flags": {
                m: {
                    c: sum(flag_of(r, m, c) in (sc.FLAGGED, sc.REVIEW_CANDIDATE) for r in rs)
                    for c in (*codes, "T9")
                }
                for m in sc.MODELS
            },
            "t8_candidates": sum(r["t8_metrics_only"] == sc.REVIEW_CANDIDATE for r in rs),
        }

    return {
        "aliases_analyzed": len(ok),
        "aliases_stopped": [r["alias"] for r in results if r["status"] != "analyzed"],
        "flag_counts_per_model": per_model,
        "t8_review_candidates": sum(r["t8_metrics_only"] == sc.REVIEW_CANDIDATE for r in ok),
        "components_total": {k: sum(r["components"][k] for r in ok) for k in ("gt", *sc.MODELS)},
        "matching_group_types": match_types,
        "groups": {g: group_pattern(g) for g in sc.GROUP_ORDER},
        "by_spacing_group": by("spacing_group"),
        "by_adc_unit_group": by("adc_unit_group"),
        "maps_written": sum(r.get("maps", 0) for r in ok),
        "figures_written": sum(r.get("figures", 0) for r in ok),
        "note": "Counts of rule outputs on selected cases; no cause, diagnosis or GT judgement.",
    }
