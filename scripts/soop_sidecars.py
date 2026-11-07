"""SOOP 1.1.2 public JSON-sidecar collection and aggregation (K4). Text files only, no images.

  python scripts/soop_sidecars.py plan       # tree + eligible list + 1-byte range check of URLs
  python scripts/soop_sidecars.py download   # sidecars (+ dwi.bval) -> .part -> verify -> rename
  python scripts/soop_sidecars.py retry      # ONE attempt per file that failed preflight
  python scripts/soop_sidecars.py aggregate  # aggregates only -> results/soop_compatibility/v1

Files are pinned to snapshot 1.1.2 (OpenNeuro snapshot URL -> S3 object with versionId) and are
verified against the git blob SHA-1 of the 1.1.2 tree. No automatic retry (``retry`` is a
single, explicit, recorded attempt per failed file and refuses to run twice), no overwrite.
The console and the aggregate outputs never contain a subject ID
(internal lists stay under ``_internal``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai"))

from neuroscan_ai.datasets.soop import (  # noqa: E402
    ADC_CONTENT_RULE_VERSION,
    ADC_GROUP_RULE_VERSION,
    subject_paths,
)
from neuroscan_ai.datasets.soop_metadata import aggregate, eligible_subjects, parse_bval  # noqa: E402

TAG, COMMIT = "1.1.2", "a62344db73f18f384a95b452f4a1a423087e059b"
API = "https://api.github.com/repos/OpenNeuroDatasets/ds004889/git"
SNAP = f"https://openneuro.org/crn/datasets/ds004889/snapshots/{TAG}/files/"
UA = {"User-Agent": "neuroscan-metadata-check"}
KINDS = ("trace_json", "adc_json")
DATASET_FILES = ("dwi.bval",)  # dwi.bvec not needed for this analysis

META = REPO_ROOT / "data/raw/soop_meta_v1_1_2"
INTERNAL = META / "_internal"
PILOT = REPO_ROOT / "data/raw/soop_pilot"
RESULTS = REPO_ROOT / "results/soop_compatibility/v1"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read())


def _ignored(path: Path) -> bool:
    rel = (path / "probe.json").relative_to(REPO_ROOT)
    return subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=REPO_ROOT).returncode == 0


def _write_new(path: Path, obj: object) -> None:
    if path.exists():
        sys.exit(f"REFUSED: {path.relative_to(REPO_ROOT).as_posix()} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _pilot_subjects() -> set[str]:
    t = json.loads((PILOT / "_internal/aliases.json").read_text(encoding="utf-8"))
    return {v["subject_id"] for v in t["aliases"].values()}


def _range(url: str) -> dict:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read(1)  # discarded
            cr = r.headers.get("Content-Range") or ""
            return {
                "status": r.status,
                "total": int(cr.rsplit("/", 1)[-1]) if "/" in cr else None,
                "pinned": "versionId=" in r.geturl(),
            }
    except Exception as e:  # noqa: BLE001
        return {"status": getattr(e, "code", type(e).__name__)}


def plan() -> int:
    for d in (META, RESULTS):
        if not _ignored(d):
            print(f"REFUSED: {d.relative_to(REPO_ROOT).as_posix()} is not git-ignored")
            return 2
    commit = _get_json(f"{API}/refs/tags/{TAG}")["object"]["sha"]
    if commit != COMMIT:
        print("REFUSED: tag 1.1.2 does not point to the recorded commit")
        return 2
    tree_sha = _get_json(f"{API}/commits/{COMMIT}")["tree"]["sha"]
    tree = _get_json(f"{API}/trees/{tree_sha}?recursive=1")
    if tree.get("truncated"):
        print("REFUSED: tree listing truncated")
        return 2
    blobs = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    eligible, missing = eligible_subjects(blobs)
    pilot = _pilot_subjects()
    jobs = []
    for s in eligible:
        if s in pilot:
            continue
        for k in KINDS:
            rel = subject_paths(s)[k]
            e = blobs[rel]
            if e["mode"] != "100644":
                print("REFUSED: a sidecar is not a plain git blob")
                return 2
            jobs.append({"path": rel, "kind": k, "git_sha1": e["sha"], "size": e["size"]})
    for rel in DATASET_FILES:
        if rel in blobs:
            e = blobs[rel]
            jobs.append({"path": rel, "kind": "dataset", "git_sha1": e["sha"], "size": e["size"]})
    for j in jobs:
        j["url"] = SNAP + j["path"].replace("/", ":")
    with cf.ThreadPoolExecutor(8) as ex:
        for j, r in zip(jobs, ex.map(lambda j: _range(j["url"]), jobs), strict=True):
            j["preflight"] = r
    ok = [j for j in jobs if j["preflight"].get("status") == 206]
    size_ok = sum(j["preflight"].get("total") == j["size"] for j in ok)
    _write_new(INTERNAL / "tree_1.1.2.json", tree)
    _write_new(
        INTERNAL / "plan.json",
        {
            "tag": TAG,
            "commit": COMMIT,
            "eligible": eligible,
            "pilot_reused": sorted(pilot),
            "jobs": jobs,
        },
    )
    print(
        json.dumps(
            {
                "eligible": len(eligible),
                "missing_required_counts": dict(missing),
                "pilot_reused_subjects": len(pilot & set(eligible)),
                "files_planned": len(jobs),
                "sidecars_planned": sum(j["kind"] != "dataset" for j in jobs),
                "dataset_files_planned": [j["path"] for j in jobs if j["kind"] == "dataset"],
                "expected_bytes": sum(j["size"] for j in jobs),
                "preflight_206": len(ok),
                "preflight_size_matches_tree": size_ok,
                "preflight_pinned": sum(bool(j["preflight"].get("pinned")) for j in ok),
                "preflight_failed": len(jobs) - len(ok),
            },
            indent=1,
        )
    )
    return 0


def _download(j: dict) -> dict:
    dest = META / j["path"]
    part = dest.with_name(dest.name + ".part")
    rec = {"path": j["path"], "url": j["url"], "snapshot": TAG}
    try:
        if dest.exists() or part.exists():
            raise FileExistsError
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(j["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            rec["pinned"] = "versionId=" in r.geturl()
        fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        sha1 = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        rec.update(
            size=len(data), sha256=hashlib.sha256(data).hexdigest(), verified=sha1 == j["git_sha1"]
        )
        if not rec["verified"]:
            raise ValueError("git blob mismatch")
        os.rename(part, dest)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - recorded, never retried
        rec.update(status="failed", error=type(e).__name__)
        if part.exists():
            part.replace(part.with_name(part.name.replace(".part", ".failed")))
    rec["completed_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    return rec


def download() -> int:
    p = json.loads((INTERNAL / "plan.json").read_text(encoding="utf-8"))
    if (INTERNAL / "download_manifest.json").exists():
        print("REFUSED: download manifest exists")
        return 2
    jobs = [j for j in p["jobs"] if j["preflight"].get("status") == 206]
    with cf.ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(_download, jobs))
    _write_new(INTERNAL / "download_manifest.json", {"tag": TAG, "commit": COMMIT, "files": recs})
    ok = [r for r in recs if r["status"] == "ok"]
    print(
        json.dumps(
            {
                "attempted": len(recs),
                "ok": len(ok),
                "failed": len(recs) - len(ok),
                "bytes": sum(r["size"] for r in ok),
                "all_verified": all(r.get("verified") for r in ok),
                "all_pinned": all(r.get("pinned") for r in ok),
                "part_or_failed_files_left": sum(
                    1 for _ in META.rglob("*") if _.suffix in (".part", ".failed")
                ),
            },
            indent=1,
        )
    )
    return 0 if len(ok) == len(recs) else 3


def retry() -> int:
    p = json.loads((INTERNAL / "plan.json").read_text(encoding="utf-8"))
    if (INTERNAL / "retry_manifest.json").exists():
        print("REFUSED: retry already performed (single attempt only)")
        return 2
    jobs = [
        j
        for j in p["jobs"]
        if j["preflight"].get("status") != 206 and not (META / j["path"]).exists()
    ]
    recs = [_download(j) for j in jobs]  # sequential, one attempt each
    _write_new(INTERNAL / "retry_manifest.json", {"tag": TAG, "commit": COMMIT, "files": recs})
    ok = [r for r in recs if r["status"] == "ok"]
    print(
        json.dumps(
            {
                "retried": len(recs),
                "ok": len(ok),
                "failed": len(recs) - len(ok),
                "verified": all(r.get("verified") for r in ok),
                "pinned": all(r.get("pinned") for r in ok),
                "failure_types": sorted({r.get("error", "") for r in recs if r["status"] != "ok"}),
            },
            indent=1,
        )
    )
    return 0 if len(ok) == len(recs) else 3


def _sidecar(subject: str, kind: str, pilot: set[str]) -> dict:
    base = PILOT if subject in pilot else META
    return json.loads((base / subject_paths(subject)[kind]).read_text(encoding="utf-8"))


def aggregate_cmd() -> int:
    p = json.loads((INTERNAL / "plan.json").read_text(encoding="utf-8"))
    pilot = set(p["pilot_reused"])
    eligible = p["eligible"]
    pairs, incomplete = [], 0
    for s in eligible:
        try:
            pairs.append((_sidecar(s, "trace_json", pilot), _sidecar(s, "adc_json", pilot)))
        except (OSError, json.JSONDecodeError):
            incomplete += 1  # metadata_incomplete: kept out of every group, never guessed
    summary = aggregate(pairs, metadata_incomplete=incomplete)
    summary["eligible_by_files"] = len(eligible)
    summary["source"] = {"dataset": "OpenNeuro ds004889", "snapshot": TAG, "commit": COMMIT}
    bval = META / "dwi.bval"
    summary["dataset_level_bval"] = (
        parse_bval(bval.read_text(encoding="utf-8")) if bval.is_file() else "not present"
    )
    per_subject_bval = sum(
        1 for s in eligible if (META / s).is_dir() and any((META / s).rglob("*.bval"))
    )
    summary["per_subject_bval_files"] = per_subject_bval
    text = json.dumps(summary, indent=2)
    leaked = [s for s in eligible if re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", text)]
    if leaked:
        print(f"REFUSED: {len(leaked)} subject IDs would appear in the aggregate")
        return 2
    name = f"sidecar_summary_{ADC_GROUP_RULE_VERSION}_{ADC_CONTENT_RULE_VERSION}.json"
    _write_new(RESULTS / name, summary)
    print(f"aggregate: {summary['n_subjects']} subjects -> results/soop_compatibility/v1/{name}")
    print(f"  adc storage groups: {summary['adc_storage_groups']}")
    print(f"  adc content status: {summary['adc_content_status']}, incomplete {incomplete}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["plan", "download", "retry", "aggregate"])
    step = ap.parse_args(argv).step
    return {"plan": plan, "download": download, "retry": retry, "aggregate": aggregate_cmd}[step]()


if __name__ == "__main__":
    sys.exit(main())
