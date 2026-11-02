"""Final Test manifest: loading, hashing and the pre-flight checks that run before any test file.

Every check returns ``Check`` records instead of raising, so the dry-run can report all of them;
``blocking_failures`` decides. In final-test mode every failed check blocks. In the validation
dry-run the Git freeze checks (clean tree, frozen commit, lock tag) are reported as pending and
do not block, because the code under rehearsal is not frozen yet; all hash and policy checks
still block.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from neuroscan_ai.evaluation.comparison import ALLOWED_CONFIG_DIFFERENCES, config_diff
from neuroscan_ai.training.config import config_hash, in_channels, load_config

MANIFEST_PATH = Path("configs/final_test_v1.yaml")
CONFIRM_PHRASE = "I APPROVE THE NEUROSCAN V1 FINAL TEST"
MODEL_KEYS = ("2d", "25d")

# What the evaluation code implements. The manifest must state exactly this.
FROZEN_POLICY: dict[str, Any] = {
    "seed": 42,
    "modalities": ["dwi", "adc"],
    "context_slices": {"2d": 0, "25d": 1},
    "preprocessing_version": "v1",
    "brain_mask_policy": "skull_stripped_background",
    "brain_mask_gating": True,
    "exclude_brain_free_slices": True,
    "threshold": 0.5,
    "postprocessing": "none",
    "small_component_removal": False,
    "precision": "fp32",
    "evaluation_path": "training_validation",
    "evaluation_space": "original",
    "small_region_10pct_rule": "viewer_display_only",
}

GIT_CHECKS = ("git_clean_tree", "git_frozen_commit", "git_lock_tag")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: str | Path) -> str:
    """SHA-256 of a text file with CRLF -> LF normalised (= the bytes Git stores)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != "final_test_v1":
        raise ValueError(f"{path}: not a final_test_v1 manifest")
    return data


def load_model_configs(manifest: dict[str, Any], repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        k: load_config(repo_root / manifest["models"][k]["training_config"]) for k in MODEL_KEYS
    }


# --- artefact hashes ----------------------------------------------------------------------------


def _hash_check(name: str, path: Path, expected: str, text: bool) -> Check:
    if not path.is_file():
        return Check(name, False, f"missing file {path.as_posix()}")
    actual = sha256_text(path) if text else sha256_file(path)
    return Check(name, actual == expected, f"expected {expected[:12]}…, actual {actual[:12]}…")


def check_artifacts(manifest: dict[str, Any], repo_root: Path) -> list[Check]:
    out = []
    for k in MODEL_KEYS:
        m = manifest["models"][k]
        out.append(
            _hash_check(
                f"checkpoint_sha256_{k}",
                repo_root / m["checkpoint"],
                m["checkpoint_sha256"],
                text=False,
            )
        )
        out.append(
            _hash_check(
                f"config_sha256_{k}",
                repo_root / m["training_config"],
                m["training_config_sha256"],
                text=True,
            )
        )
    for name, c in manifest["configs"].items():
        out.append(_hash_check(f"config_sha256_{name}", repo_root / c["path"], c["sha256"], True))
    s = manifest["split"]
    out.append(_hash_check("split_sha256", repo_root / s["path"], s["sha256"], text=True))
    return out


def check_checkpoint_info(manifest: dict[str, Any], repo_root: Path) -> list[Check]:
    """The checkpoint's recorded config hash / epoch / channels match the manifest (CPU load)."""
    import torch

    out = []
    for k in MODEL_KEYS:
        m = manifest["models"][k]
        path = repo_root / m["checkpoint"]
        if not path.is_file():
            out.append(Check(f"checkpoint_info_{k}", False, "checkpoint missing"))
            continue
        info = torch.load(path, map_location="cpu", weights_only=False).get("info", {})
        ok = (
            info.get("config_hash") == m["config_hash"]
            and info.get("epoch") == m["best_epoch"]
            and info.get("in_channels") == m["in_channels"]
        )
        out.append(
            Check(
                f"checkpoint_info_{k}",
                ok,
                f"config_hash {info.get('config_hash')}, epoch {info.get('epoch')}, "
                f"in_channels {info.get('in_channels')}",
            )
        )
    return out


# --- frozen policy ------------------------------------------------------------------------------


def check_policy(manifest: dict[str, Any], cfgs: dict[str, dict[str, Any]]) -> list[Check]:
    out = []
    pol = manifest.get("policy", {})
    diffs = {k: (pol.get(k), v) for k, v in FROZEN_POLICY.items() if pol.get(k) != v}
    out.append(
        Check(
            "manifest_policy_matches_code",
            not diffs,
            "ok" if not diffs else f"manifest vs code: {diffs}",
        )
    )
    norm = pol.get("normalization")
    for k in MODEL_KEYS:
        cfg, m = cfgs[k], manifest["models"][k]
        d, inf = cfg["data"], cfg["inference"]
        problems = []
        if config_hash(cfg) != m["config_hash"]:
            problems.append(f"config_hash {config_hash(cfg)} != {m['config_hash']}")
        if inf.get("threshold") != FROZEN_POLICY["threshold"]:
            problems.append(f"threshold {inf.get('threshold')}")
        if inf.get("brain_mask_gating") is not True:
            problems.append("brain_mask_gating not true")
        if cfg.get("postprocessing", "none") not in (None, "none") or "postprocessing" in inf:
            problems.append("post-processing configured")
        if cfg["training"].get("precision") != FROZEN_POLICY["precision"]:
            problems.append(f"precision {cfg['training'].get('precision')}")
        if cfg.get("seed") != FROZEN_POLICY["seed"]:
            problems.append(f"seed {cfg.get('seed')}")
        if d.get("context_slices") != FROZEN_POLICY["context_slices"][k]:
            problems.append(f"context_slices {d.get('context_slices')}")
        if list(d.get("modalities", [])) != FROZEN_POLICY["modalities"]:
            problems.append(f"modalities {d.get('modalities')}")
        if d.get("preprocessing_version") != FROZEN_POLICY["preprocessing_version"]:
            problems.append(f"preprocessing_version {d.get('preprocessing_version')}")
        if d.get("brain_mask_policy") != FROZEN_POLICY["brain_mask_policy"]:
            problems.append(f"brain_mask_policy {d.get('brain_mask_policy')}")
        if d.get("exclude_brain_free_slices") is not True:
            problems.append("exclude_brain_free_slices not true")
        if norm is not None and d.get("normalization") != norm:
            problems.append("normalization differs from manifest")
        if Path(d.get("split_file", "")).as_posix() != manifest["split"]["path"]:
            problems.append(f"split_file {d.get('split_file')}")
        if in_channels(cfg) != m["in_channels"]:
            problems.append(f"in_channels {in_channels(cfg)}")
        out.append(Check(f"training_config_policy_{k}", not problems, "; ".join(problems) or "ok"))
    diff = config_diff(cfgs["2d"], cfgs["25d"])
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFERENCES)
    out.append(
        Check(
            "2d_vs_25d_only_context_differs",
            not unexpected,
            "ok" if not unexpected else f"unexpected differences: {unexpected}",
        )
    )
    return out


# --- Git freeze ---------------------------------------------------------------------------------


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def check_git(manifest: dict[str, Any], repo_root: Path, manifest_rel: str) -> list[Check]:
    out = []
    head_p = _git(repo_root, "rev-parse", "HEAD")
    head = head_p.stdout.strip() if head_p.returncode == 0 else ""
    status = _git(repo_root, "status", "--porcelain")
    clean = status.returncode == 0 and status.stdout.strip() == ""
    out.append(Check("git_clean_tree", clean, "clean" if clean else "working tree has changes"))

    frozen = (manifest.get("git") or {}).get("frozen_commit")
    if not frozen:
        out.append(Check("git_frozen_commit", False, "frozen_commit not set in the manifest"))
    else:
        full = _git(repo_root, "rev-parse", "--verify", f"{frozen}^{{commit}}")
        frozen_full = full.stdout.strip() if full.returncode == 0 else ""
        if not frozen_full:
            out.append(Check("git_frozen_commit", False, "frozen_commit not found"))
        else:
            ancestor = _git(repo_root, "merge-base", "--is-ancestor", frozen_full, head)
            changed = _git(repo_root, "diff", "--name-only", frozen_full, head).stdout.split()
            extra = sorted(set(changed) - {manifest_rel})
            ok = ancestor.returncode == 0 and not extra
            detail = (
                f"HEAD {head[:12]} descends from {frozen_full[:12]}, differs only in the manifest"
                if ok
                else f"HEAD {head[:12]} vs frozen {frozen_full[:12]}: "
                f"ancestor={ancestor.returncode == 0}, other changed files={extra}"
            )
            out.append(Check("git_frozen_commit", ok, detail))

    tag = (manifest.get("git") or {}).get("lock_tag")
    tag_p = _git(repo_root, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}") if tag else None
    tag_commit = tag_p.stdout.strip() if tag_p is not None and tag_p.returncode == 0 else ""
    if not tag_commit:
        out.append(Check("git_lock_tag", False, f"tag {tag} does not exist"))
    else:
        out.append(
            Check(
                "git_lock_tag",
                tag_commit == head,
                f"tag {tag} -> {tag_commit[:12]}, HEAD {head[:12]}",
            )
        )
    return out


def head_commit(repo_root: Path) -> str | None:
    p = _git(repo_root, "rev-parse", "HEAD")
    return p.stdout.strip() if p.returncode == 0 else None


def blocking_failures(checks: list[Check], mode: str) -> list[Check]:
    if mode == "final-test":
        return [c for c in checks if not c.ok]
    return [c for c in checks if not c.ok and c.name not in GIT_CHECKS]
