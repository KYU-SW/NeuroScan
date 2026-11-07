"""Run reproducibility manifests: synthetic files, a throwaway git repo, fake torch / nvidia-smi.
No MRI, no model, no real dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuroscan_ai.reproducibility import (
    CASE_KEY_POLICY,
    COMPATIBLE,
    EXACT,
    MISMATCH,
    SCHEMA,
    UNAVAILABLE,
    ManifestError,
    PathRoots,
    RunSpec,
    build_post_run,
    build_pre_run,
    case_keys,
    check_schema,
    compare_manifests,
    is_linked,
    write_case_key_map,
    write_manifest,
)
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    file_hashes,
    sha256_file,
    sha256_text_lf,
)
from neuroscan_ai.reproducibility.manifest import sanitize_path

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]


def _git_repo(tmp: Path) -> Path:
    repo = tmp / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "cfg.yaml").write_text("threshold: 0.5\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "data/\nmodels/\nout/\n", encoding="utf-8"
    )  # like the real repo
    (repo / "requirements-lock.txt").write_bytes(b"numpy==2.0\r\nscipy==1.0\r\n")
    subprocess.run(
        [*GIT, "-C", str(repo), "add", "cfg.yaml", ".gitignore", "requirements-lock.txt"],
        check=True,
    )
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    subprocess.run([*GIT, "-C", str(repo), "tag", "lock-synthetic"], check=True)
    return repo


def _torch(cuda: bool = False, tf32_conv: bool = True, det: bool = True):
    return SimpleNamespace(
        __version__="9.9.9+synthetic",
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: "Synthetic GPU"),
        version=SimpleNamespace(cuda="12.9" if cuda else None),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                is_available=lambda: cuda,
                version=lambda: 90000,
                deterministic=det,
                benchmark=not det,
                allow_tf32=tf32_conv,
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        ),
        are_deterministic_algorithms_enabled=lambda: det,
        is_deterministic_algorithms_warn_only_enabled=lambda: True,
    )


def _smi(output="Synthetic GPU, 600.00\n", rc=0):
    def run(argv, **kw):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, rc, stdout=output, stderr="")
        return subprocess.run(argv, **kw)

    return run


VERSIONS = {
    "torch": "9.9.9",
    "monai": "1.0",
    "numpy": "2.0",
    "nibabel": "5.0",
    "scipy": "1.0",
}.__getitem__


@pytest.fixture
def world(tmp_path):
    repo = _git_repo(tmp_path)
    data = repo / "data"
    models = repo / "models"
    data.mkdir()
    models.mkdir()
    (data / "split.json").write_bytes(b'{"val": ["case-a", "case-b"]}\r\n')
    (models / "best.pt").write_bytes(b"synthetic-checkpoint")
    for a in ("case-a", "case-b"):
        (data / f"{a}_dwi.nii.gz").write_bytes(b"dwi-" + a.encode())
    return SimpleNamespace(repo=repo, data=data, models=models, roots=PathRoots(repo, models, data))


def _spec(w, **kw):
    s = RunSpec(
        run_id="synthetic-run",
        stage="v2-validation",
        purpose="unit test",
        dataset_id="synthetic",
        split_name="val",
        aliases=["case-a", "case-b"],
        config=w.repo / "cfg.yaml",
        split=w.data / "split.json",
        checkpoint=w.models / "best.pt",
        policies={"threshold": "fixed_0.5", "brain_mask": "synthetic_v1"},
        model={
            "architecture_id": "unet_synthetic",
            "activation": "sigmoid",
            "fixed_threshold": 0.5,
        },
        input_files={a: {"dwi": w.data / f"{a}_dwi.nii.gz"} for a in ("case-a", "case-b")},
        access={"val_cases": 2, "allowed_splits": ["val"]},
    )
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _pre(w, torch=None, **kw):
    t = torch or _torch()
    return build_pre_run(
        _spec(w, **kw),
        roots=w.roots,
        source_control=capture_source_control(w.repo, "lock-synthetic"),
        environment=_env(w, t),
        determinism=capture_determinism(
            t, seed=42, precision="fp32", amp=False, workers=0, batch_size=8
        ),
    )


def _env(w, t):
    lock = w.repo / "requirements-lock.txt"
    return capture_environment(torch_module=t, run=_smi(), versions=VERSIONS, lock_file=lock)


ALIASES = ["case-a", "case-b"]


def _post(w, pre, prob=b"p"):
    out = w.repo / "out"
    out.mkdir(exist_ok=True)
    (out / "case-a_prob.nii.gz").write_bytes(prob)
    t = _torch()
    return build_post_run(
        pre,
        aliases=ALIASES,
        started_utc="2026-01-01T00:00:00+00:00",
        ended_utc="2026-01-01T00:01:00+00:00",
        status="success",
        exit_code=0,
        cases_succeeded=2,
        cases_failed=0,
        outputs={
            "probability_maps": {"case-a": out / "case-a_prob.nii.gz"},
            "aggregate_results": {"summary": out / "case-a_prob.nii.gz"},
        },
        environment=_env(w, t),
        determinism=capture_determinism(
            t, seed=42, precision="fp32", amp=False, workers=0, batch_size=8
        ),
        access=pre["access"],
    )


# ---------------------------------------------------------------- 1-3 git
def test_01_clean_git(world):
    sc = capture_source_control(world.repo, "lock-synthetic")
    assert sc["working_tree_clean"] is True and sc["tracked_diff_present"] is False


def test_02_dirty_git_recorded(world):
    (world.repo / "cfg.yaml").write_text("threshold: 0.4\n", encoding="utf-8")
    (world.repo / "new.txt").write_text("x", encoding="utf-8")
    sc = capture_source_control(world.repo)
    assert (
        sc["working_tree_clean"] is False
        and sc["tracked_diff_present"]
        and sc["untracked_files_present"]
    )


def test_03_git_commit_and_tag(world):
    sc = capture_source_control(world.repo, "lock-synthetic")
    head = subprocess.run(
        ["git", "-C", str(world.repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert sc["git_commit"] == head == sc["lock_tag_target"] and sc["branch"] == "main"


# ---------------------------------------------------------------- 4-8 hashes
def test_04_config_hash(world):
    pre = _pre(world)
    assert pre["configuration"]["config"]["sha256"] == sha256_file(world.repo / "cfg.yaml")


def test_05_06_canonical_lf(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"x: 1\r\ny: 2\r\n")
    b.write_bytes(b"x: 1\ny: 2\n")
    assert sha256_file(a) != sha256_file(b) and sha256_text_lf(a) == sha256_text_lf(b)


def test_07_split_hash(world):
    s = _pre(world)["configuration"]["split"]
    assert s == {"path": "<DATASET>/split.json", **file_hashes(world.data / "split.json")}
    assert s["sha256"] != s["sha256_lf"]


def test_08_checkpoint_hash(world):
    c = _pre(world)["model"]["checkpoint"]
    assert c["path"] == "<CHECKPOINT>/best.pt" and c["sha256"] == sha256_file(
        world.models / "best.pt"
    )


# ---------------------------------------------------------------- 9-14 environment / determinism
def test_09_package_versions(world):
    env = _pre(world)["environment"]
    assert env["packages"] == {
        "torch": "9.9.9",
        "monai": "1.0",
        "numpy": "2.0",
        "nibabel": "5.0",
        "scipy": "1.0",
    }


def test_10_no_cuda():
    env = capture_environment(torch_module=_torch(cuda=False), run=_smi(), versions=VERSIONS)
    assert (
        env["cuda_runtime"] is None
        and env["gpu_driver_version"] is None
        and env["torch_cuda_available"] is False
    )


def test_11_cuda_present():
    env = capture_environment(torch_module=_torch(cuda=True), run=_smi(), versions=VERSIONS)
    assert (env["cuda_runtime"], env["cudnn"], env["gpu_name"], env["gpu_driver_version"]) == (
        "12.9",
        90000,
        "Synthetic GPU",
        "600.00",
    )


def test_12_driver_missing():
    env = capture_environment(
        torch_module=_torch(cuda=True), run=_smi(output="", rc=9), versions=VERSIONS
    )
    assert env["gpu_driver_version"] is None and env["gpu_name"] == "Synthetic GPU"


def test_13_tf32_recorded():
    d = capture_determinism(
        _torch(tf32_conv=True), seed=1, precision="fp32", amp=False, workers=0, batch_size=4
    )
    assert d["tf32_convolution"] is True and d["tf32_matmul"] is False


def test_14_deterministic_recorded():
    d = capture_determinism(
        _torch(det=True), seed=42, precision="fp32", amp=False, workers=0, batch_size=4
    )
    assert (
        d["torch_deterministic_algorithms"]
        and d["cudnn_deterministic"]
        and not d["cudnn_benchmark"]
    )
    assert d["seed"] == 42 and d["batch_size"] == 4


# ---------------------------------------------------------------- 15-17 pre / post
def test_15_pre_run(world):
    pre = _pre(world)
    assert pre["kind"] == "pre_run" and pre["identity"]["case_count"] == 2
    assert pre["access"]["val_cases"] == 2 and pre["access"]["disallowed_split_access"] is False
    assert pre["configuration"]["policies"]["threshold_policy_id"] == "fixed_0.5"


def test_16_17_post_run_linked(world):
    pre = _pre(world)
    post = _post(world, pre)
    assert post["kind"] == "post_run" and post["run"]["status"] == "success"
    assert post["outputs"]["probability_maps"]["case_0001"] == sha256_file(
        world.repo / "out/case-a_prob.nii.gz"
    )
    assert is_linked(pre, post)
    tampered = copy.deepcopy(pre)
    tampered["identity"]["purpose"] = "changed"
    assert not is_linked(tampered, post)


def test_disallowed_split_refused(world):
    with pytest.raises(ManifestError):
        _pre(world, access={"val_cases": 2, "test_cases": 1, "allowed_splits": ["val"]})


# ---------------------------------------------------------------- 18-26 comparison levels
def test_18_exact_match(world):
    pre = _pre(world)
    assert compare_manifests(pre, copy.deepcopy(pre), cpu_only_ok=True)["level"] == EXACT


@pytest.mark.parametrize(
    "path,level",
    [
        (("source_control", "git_commit"), COMPATIBLE),  # 19
        (("environment", "packages", "torch"), COMPATIBLE),  # 20
        (("environment", "cuda_runtime"), COMPATIBLE),  # 21
        (("environment", "gpu_driver_version"), COMPATIBLE),  # 22
        (("configuration", "config", "sha256_lf"), MISMATCH),  # 23
        (("model", "checkpoint", "sha256"), MISMATCH),  # 24
        (("configuration", "split", "sha256_lf"), MISMATCH),  # 25
        (("determinism", "tf32_convolution"), COMPATIBLE),
        (("environment", "cudnn"), COMPATIBLE),
        (("environment", "gpu_name"), COMPATIBLE),
        (("environment", "package_lock_sha256_lf"), COMPATIBLE),
        (("inputs", "alias_list_sha256"), MISMATCH),
    ],
)
def test_19_25_differences_prevent_exact(world, path, level):
    a = _pre(world, torch=_torch(cuda=True))  # CUDA fields present on both sides
    b = copy.deepcopy(a)
    node = b
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = "different"
    res = compare_manifests(a, b, cpu_only_ok=True)
    assert res["level"] == level and res["level"] != EXACT


def test_26_output_hash_mismatch(world):
    pre = _pre(world)
    a = _post(world, pre, prob=b"p1")
    b = _post(world, pre, prob=b"p2")
    assert compare_manifests(a, b, cpu_only_ok=True)["level"] == MISMATCH


def test_unavailable_when_unknown_on_both_sides(world):
    a = _pre(world)
    b = copy.deepcopy(a)
    a["source_control"]["git_commit"] = b["source_control"]["git_commit"] = None
    assert compare_manifests(a, b, cpu_only_ok=True)["level"] == UNAVAILABLE


def test_dirty_tree_not_exact(world):
    a = _pre(world)
    b = copy.deepcopy(a)
    b["source_control"]["working_tree_clean"] = False
    assert compare_manifests(a, b, cpu_only_ok=True)["level"] == COMPATIBLE


# ---------------------------------------------------------------- 27-32 privacy and writing
def test_27_original_id_refused(world):
    with pytest.raises(ManifestError):
        _pre(world, aliases=["sub-strokecase9991"], input_files={})
    with pytest.raises(ManifestError):
        _pre(world, purpose="check sub-zz123")


def test_28_absolute_paths_sanitised(world, tmp_path):
    assert (
        sanitize_path(world.repo / "configs" / "x.yaml", world.roots) == "<WORKDIR>/configs/x.yaml"
    )
    assert sanitize_path(world.data / "a" / "b.nii.gz", world.roots) == "<DATASET>/a/b.nii.gz"
    with pytest.raises(ManifestError):
        sanitize_path(tmp_path / "elsewhere.txt", world.roots)
    text = json.dumps(_pre(world))
    assert str(world.repo) not in text and world.repo.as_posix() not in text


@pytest.mark.parametrize(
    "secret", ["ghp_" + "A" * 30, "password=hunter2x", "Bearer abcdefghijklmnopqrstu"]
)
def test_29_credentials_refused(world, secret):
    with pytest.raises(ManifestError):
        _pre(world, purpose=secret)


def test_29b_secret_env_value_refused(world, monkeypatch):
    monkeypatch.setenv("SYNTHETIC_API_TOKEN", "zz-secret-value-123")
    with pytest.raises(ManifestError):
        _pre(world, purpose="uses zz-secret-value-123")


def test_30_aliases_never_in_plain_text(world):
    pre = _pre(world)  # input files present for both aliases
    post = _post(world, pre)  # per-case output present
    for m in (pre, post):
        text = json.dumps(m)
        assert all(a not in text for a in ALIASES)
    assert set(pre["inputs"]["file_sha256"]) == {"case_0001", "case_0002"}
    assert pre["inputs"]["file_sha256"]["case_0001"] == {
        "dwi_sha256": sha256_file(world.data / "case-a_dwi.nii.gz")
    }
    assert pre["inputs"]["case_key_policy"] == CASE_KEY_POLICY == "sorted_alias_position_v1"
    assert set(post["outputs"]["probability_maps"]) == {"case_0001"}


def test_30b_alias_in_recorded_path_replaced(world):
    sub = world.data / "case-a"
    sub.mkdir()
    (sub / "split.json").write_text("{}", encoding="utf-8")
    pre = _pre(world, split=sub / "split.json")
    assert pre["configuration"]["split"]["path"] == "<DATASET>/case_0001/split.json"
    assert "case-a" not in json.dumps(pre)


def test_30c_case_key_rule():
    assert case_keys(["b-02", "a-01", "c-03"]) == {
        "a-01": "case_0001",
        "b-02": "case_0002",
        "c-03": "case_0003",
    }
    assert case_keys(["c-03", "a-01", "b-02"]) == case_keys(["a-01", "b-02", "c-03"])


def test_30d_alias_list_hash_depends_on_set_and_ordering_policy():
    same = alias_list_sha256(["b", "a"], CASE_KEY_POLICY)
    assert same == alias_list_sha256(["a", "b"], CASE_KEY_POLICY)  # same set, same rule
    assert same != alias_list_sha256(["a", "c"], CASE_KEY_POLICY)  # different set
    other_rule = hashlib.sha256(b"reverse_sorted_v0\nb\na").hexdigest()
    assert same != other_rule  # the ordering rule is part of the hashed text
    with pytest.raises(ValueError):
        alias_list_sha256(["a"], "unsorted")


def test_30e_post_run_refuses_different_alias_list(world):
    pre = _pre(world)
    with pytest.raises(ManifestError):
        build_post_run(
            pre,
            aliases=["case-a", "case-z"],
            started_utc="s",
            ended_utc="e",
            status="success",
            exit_code=0,
            cases_succeeded=2,
            cases_failed=0,
            outputs={},
            environment={},
            determinism={},
            access={},
        )


def test_30f_internal_map_only_in_ignored_internal_dir(tmp_path):
    with pytest.raises(ManifestError):
        write_case_key_map(tmp_path / "map.json", ALIASES, REPO)
    with pytest.raises(ManifestError):
        write_case_key_map(REPO / "results/reproducibility/map.json", ALIASES, REPO)
    probe = "results/reproducibility/_internal/case_key_map.json"
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", probe]).returncode == 0


def test_30g_write_manifest_refuses_alias_leak(world, tmp_path):
    pre = _pre(world)
    leaked = copy.deepcopy(pre)
    leaked["identity"]["purpose"] = "case-a only"
    with pytest.raises(ManifestError):
        write_manifest(tmp_path / "m.json", leaked, aliases=ALIASES)


# ---------------------------------------------------------------- package lock
def test_lock_same_hash_exact(world):
    a = _pre(world)
    assert a["environment"]["package_lock_sha256_lf"] is not None
    assert compare_manifests(a, copy.deepcopy(a), cpu_only_ok=True)["level"] == EXACT


def test_lock_missing_is_unavailable(world):
    a = _pre(world)
    b = copy.deepcopy(a)
    b["environment"]["package_lock_sha256_lf"] = None
    res = compare_manifests(a, b, cpu_only_ok=True)
    assert res["level"] == UNAVAILABLE and "environment.package_lock_sha256_lf" in res["missing"]


def test_lock_line_endings_vs_content(tmp_path):
    crlf, lf, other = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"
    crlf.write_bytes(b"numpy==2.0\r\nscipy==1.0\r\n")
    lf.write_bytes(b"numpy==2.0\nscipy==1.0\n")
    other.write_bytes(b"numpy==2.1\nscipy==1.0\n")
    env = [
        capture_environment(torch_module=None, versions=VERSIONS, lock_file=f)
        for f in (crlf, lf, other)
    ]
    assert env[0]["package_lock_sha256_lf"] == env[1]["package_lock_sha256_lf"]
    assert env[1]["package_lock_sha256_lf"] != env[2]["package_lock_sha256_lf"]


# ---------------------------------------------------------------- schema
def test_schema_exact_accepted(world):
    pre = _pre(world)
    assert pre["schema"] == SCHEMA == "neuroscan-run-manifest-v1"
    check_schema(pre, "pre_run")
    assert is_linked(pre, _post(world, pre))


@pytest.mark.parametrize("bad", [None, "neuroscan-run-manifest-v2", ""])
def test_schema_missing_or_other_refused(world, bad, tmp_path):
    pre = _pre(world)
    other = copy.deepcopy(pre)
    if bad is None:
        del other["schema"]
    else:
        other["schema"] = bad
    with pytest.raises(ManifestError):
        compare_manifests(pre, other)
    with pytest.raises(ManifestError):
        write_manifest(tmp_path / "m.json", other)
    with pytest.raises(ManifestError):
        _post(world, other)


def test_schema_pre_post_mismatch_refused(world):
    pre = _pre(world)
    post = _post(world, pre)
    post["schema"] = "neuroscan-run-manifest-v0"
    with pytest.raises(ManifestError):
        is_linked(pre, post)
    with pytest.raises(ManifestError):
        compare_manifests(pre, post)


def test_schema_documented():
    doc = (REPO / "docs/reproducibility_protocol.md").read_text(encoding="utf-8")
    assert "`neuroscan-run-manifest-v1`" in doc and "sorted_alias_position_v1" in doc


def test_31_no_overwrite(world, tmp_path):
    pre = _pre(world)
    write_manifest(tmp_path / "m.json", pre)
    with pytest.raises(ManifestError):
        write_manifest(tmp_path / "m.json", pre)


def test_32_output_root_git_ignored():
    p = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", "results/reproducibility/r/pre_run.json"]
    )
    assert p.returncode == 0


def test_32b_cli_refuses_output_outside_root(tmp_path):
    from scripts.capture_run_manifest import main

    argv = [
        "capture",
        "--run-id",
        "r",
        "--stage",
        "s",
        "--purpose",
        "p",
        "--dataset-id",
        "d",
        "--split-name",
        "val",
        "--split",
        str(REPO / "pyproject.toml"),
        "--config",
        str(REPO / "pyproject.toml"),
        "--out",
        str(tmp_path / "m.json"),
    ]
    assert main(argv) == 2 and not any(tmp_path.iterdir())


# ------------------------------------------------- 33-35 no data, no model, no v1 change
def test_33_35_no_data_model_or_v1_imports():
    prefixes = (
        "torch",
        "monai",
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.final_test",
    )
    code = (
        "import sys; sys.path.insert(0, 'ai'); import neuroscan_ai.reproducibility\n"
        f"print([m for m in sys.modules if m.startswith({prefixes!r})])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (REPO / "ai/neuroscan_ai/reproducibility").glob("*.py")
    )
    for forbidden in (
        "load_state_dict",
        "nib.load",
        "get_fdata",
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
    ):
        assert forbidden not in src
