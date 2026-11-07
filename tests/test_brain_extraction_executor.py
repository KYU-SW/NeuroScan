"""SynthStrip Docker executor — fake process runner only (no Docker, no MRI, no network)."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.brain_extraction.audit import find_host_paths, guard_public, sanitize_paths
from neuroscan_ai.brain_extraction.executors import (
    INPUT_NAME,
    MASK_NAME,
    PILOT_POLICY,
    DockerExecutionError,
    DockerSynthStripExecutor,
    _approved_workdir,
    _check_args_safe,
    _truncate,
    build_docker_argv,
    expected_tool_argv,
    pilot_spec,
    validate_docker_argv,
)
from neuroscan_ai.brain_extraction.interfaces import STATUS_FAIL, STATUS_PASS, RunRequest
from neuroscan_ai.brain_extraction.runners import HDBetSpec, SynthStripRunner

REPO = Path(__file__).resolve().parents[1]
DIGEST = "sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06"
SHAPE = (24, 26, 12)
AFF = np.array(
    [[1.5, 0.05, 0, -18.0], [-0.05, 1.5, 0.1, -20.0], [0, -0.2, 3.0, -15.0], [0, 0, 0, 1]]
)


def _ell(r):
    g = np.indices(SHAPE).astype(float)
    c = [(s - 1) / 2 for s in SHAPE]
    return sum(((g[i] - c[i]) / (r * s / 2)) ** 2 for i, s in enumerate(SHAPE)) <= 1


BRAIN, HEAD = _ell(0.7), _ell(0.92)
GT = np.zeros(SHAPE, bool)
GT[11:13, 12:14, 5:7] = True


class FakeDocker:
    """Stands in for subprocess.run; records every argv; never starts a process."""

    def __init__(self, *, exit_code=0, write_mask=True, timeout=False, leftover=0,
                 image_id=DIGEST, engine=True, stdout="ok", stderr=""):  # fmt: skip
        self.calls: list[list[str]] = []
        self.exit_code, self.write_mask, self.timeout = exit_code, write_mask, timeout
        self.leftover, self.image_id, self.engine = leftover, image_id, engine
        self.stdout, self.stderr = stdout, stderr

    def __call__(self, argv, **kw):
        assert kw.get("shell") is False and isinstance(argv, list)
        self.calls.append(list(argv))
        ok = lambda out="": subprocess.CompletedProcess(argv, 0, out, "")  # noqa: E731
        if argv[:2] == ["docker", "info"]:
            return ok("29.7.2") if self.engine else subprocess.CompletedProcess(argv, 1, "", "x")
        if argv[:3] == ["docker", "image", "inspect"]:
            return ok(self.image_id + "\n")
        if argv[:2] == ["docker", "ps"]:
            return ok("abc\n" * self.leftover)
        if argv[:2] == ["docker", "rm"]:
            return ok("")
        if argv[:2] == ["docker", "run"] and "--entrypoint" in argv:
            return ok(f"{PILOT_POLICY.model_sha256}  {PILOT_POLICY.model_path_in_image}\n")
        if argv[:2] == ["docker", "run"]:
            if self.timeout:
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            host = argv[argv.index("-v") + 1].rpartition(":")[0]
            if self.write_mask and self.exit_code == 0:
                src = nib.load(str(Path(host) / INPUT_NAME))
                m = nib.Nifti1Image(BRAIN.astype(np.uint8), src.affine, src.header.copy())
                m.set_data_dtype(np.uint8)
                nib.save(m, str(Path(host) / MASK_NAME))
            return subprocess.CompletedProcess(argv, self.exit_code, self.stdout, self.stderr)
        raise AssertionError(f"unexpected command {argv[:3]}")


@pytest.fixture
def repo(tmp_path):
    """Fake repository root with the approved output root and a synthetic case."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    inp = root / "data/interim/synthetic"
    inp.mkdir(parents=True)
    data = np.where(BRAIN, 600, np.where(HEAD, 250, 0)).astype(np.int16)
    t = nib.Nifti1Image(data, AFF)
    t.set_qform(AFF, code=1)
    t.set_sform(AFF, code=1)
    nib.save(t, str(inp / "trace.nii.gz"))
    g = nib.Nifti1Image(GT.astype(np.uint8), AFF)
    g.set_qform(AFF, code=1)
    g.set_sform(AFF, code=1)
    nib.save(g, str(inp / "gt.nii.gz"))
    return root


def _run(repo, fake, alias="smoke-x-01", **kw):
    ex = DockerSynthStripExecutor(repo, run_fn=fake, **kw)
    req = RunRequest(alias, repo / "data/interim/synthetic/trace.nii.gz",
                     repo / "data/interim/synthetic/gt.nii.gz", "synthstrip", "1.8", {},
                     repo / "results/brain_extraction_pilot/_smoke")  # fmt: skip
    return SynthStripRunner(pilot_spec(), ex).run(req), ex


def _workdir(repo, name="a.part"):
    w = repo / "results/brain_extraction_pilot/_smoke/x" / name
    w.mkdir(parents=True)
    (w / INPUT_NAME).write_bytes(b"x")
    return w


# 1 approved digest -----------------------------------------------------------------------------
def test_approved_digest_run_succeeds(repo):
    fake = FakeDocker()
    res, ex = _run(repo, fake)
    assert res.status == STATUS_PASS and res.mask_path.is_file()
    run = [c for c in fake.calls if c[:2] == ["docker", "run"]][0]
    assert f"freesurfer/synthstrip@{DIGEST}" in run
    assert run[run.index("--network") + 1] == "none" and "--rm" in run
    assert run[-8:] == ["-i", "/data/tool_input.nii.gz", "-m", "/data/tool_mask.nii.gz",
                        "-b", "1", "-t", "4"]  # fmt: skip
    rec = ex.internal_records[0]
    assert rec["exit_code"] == 0 and rec["leftover_containers"] == 0 and rec["mask_present"]
    assert res.audit["execution"]["policy"]["name"] == "bm_synthstrip_1_8_cpu_pilot"


def test_pilot_policy_values_are_fixed():
    p = PILOT_POLICY
    assert (p.digest, p.device, p.threads, p.border_mm, p.no_csf, p.network) == (
        DIGEST, "cpu", 4, 1.0, False, "none")  # fmt: skip
    assert p.model_sha256 == "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"
    with pytest.raises(FrozenInstanceError):
        p.threads = 8  # frozen


# 2-8 refused configurations --------------------------------------------------------------------
def test_wrong_digest_or_changed_policy_refused(repo):
    changes = [{"digest": "sha256:" + "0" * 64}, {"threads": 8}, {"border_mm": 2.0},
               {"no_csf": True}, {"device": "gpu"}]  # fmt: skip
    for bad in (replace(PILOT_POLICY, **c) for c in changes):
        with pytest.raises(DockerExecutionError, match="approved pilot policy"):
            DockerSynthStripExecutor(repo, policy=bad, run_fn=FakeDocker())


def test_local_image_with_other_digest_refused(repo):
    res, _ = _run(repo, FakeDocker(image_id="sha256:" + "1" * 64))
    assert res.status == STATUS_FAIL and res.audit["execution_error"] == "DockerExecutionError"


def _argv(repo):
    w = _workdir(repo)
    return w, build_docker_argv(w, "neuroscan-bet-x")


def test_tag_only_image_refused(repo):
    w, a = _argv(repo)
    a[a.index(PILOT_POLICY.image)] = "freesurfer/synthstrip:1.8"
    with pytest.raises(DockerExecutionError, match="digest"):
        validate_docker_argv(a, w)


def test_missing_network_none_refused(repo):
    w, a = _argv(repo)
    i = a.index("--network")
    del a[i : i + 2]
    with pytest.raises(DockerExecutionError, match="network"):
        validate_docker_argv(a, w)
    w, a = _argv(repo) if False else (w, build_docker_argv(w, "n"))
    a[a.index("--network") + 1] = "host"
    with pytest.raises(DockerExecutionError, match="network"):
        validate_docker_argv(a, w)


def test_gpu_option_refused(repo):
    w, a = _argv(repo)
    with pytest.raises(DockerExecutionError, match="forbidden"):
        validate_docker_argv(a[:2] + ["--gpus", "all"] + a[2:], w)
    ex = DockerSynthStripExecutor(repo, run_fn=FakeDocker())
    with pytest.raises(DockerExecutionError, match="fixed policy"):
        ex(expected_tool_argv(w) + ["-g"], w)
    with pytest.raises(DockerExecutionError, match="fixed policy"):
        validate_docker_argv(a + ["-g"], w)


@pytest.mark.parametrize("extra", [["-v", "/tmp:/x"], ["--mount", "type=bind,src=/,dst=/h"],
                                   ["--volume", "/etc:/e"]])  # fmt: skip
def test_additional_mount_refused(repo, extra):
    w, a = _argv(repo)
    with pytest.raises(DockerExecutionError):
        validate_docker_argv(a[:2] + extra + a[2:], w)


def test_repository_and_raw_data_mount_refused(repo):
    with pytest.raises(DockerExecutionError, match="repository"):
        _approved_workdir(repo, repo, ("results/brain_extraction_pilot",))
    raw = repo / "data/raw/x.part"
    raw.mkdir(parents=True)
    (raw / INPUT_NAME).write_bytes(b"x")
    with pytest.raises(DockerExecutionError, match="approved output root"):
        _approved_workdir(raw, repo, ("results/brain_extraction_pilot",))
    w, a = _argv(repo)
    a[a.index("-v") + 1] = f"{repo}:/data"
    with pytest.raises(DockerExecutionError, match="work folder"):
        validate_docker_argv(a, w)


def test_workdir_must_be_isolated(repo):
    w = _workdir(repo)
    (w / "other.nii.gz").write_bytes(b"x")
    with pytest.raises(DockerExecutionError, match="only the tool input"):
        _approved_workdir(w, repo, ("results/brain_extraction_pilot",))
    plain = repo / "results/brain_extraction_pilot/plain"
    plain.mkdir(parents=True)
    with pytest.raises(DockerExecutionError, match=".part"):
        _approved_workdir(plain, repo, ("results/brain_extraction_pilot",))


def test_docker_socket_mount_refused(repo):
    w, a = _argv(repo)
    a[a.index("-v") + 1] = "/var/run/docker.sock:/data"
    with pytest.raises(DockerExecutionError):
        validate_docker_argv(a, w)


@pytest.mark.parametrize("flag", [["--privileged"], ["-e", "TOKEN=x"], ["--env-file", "f"],
                                  ["--cap-add", "ALL"], ["--device", "/dev/x"]])  # fmt: skip
def test_privileged_env_and_devices_refused(repo, flag):
    w, a = _argv(repo)
    with pytest.raises(DockerExecutionError, match="forbidden"):
        validate_docker_argv(a[:2] + flag + a[2:], w)


def test_missing_rm_or_pull_never_refused(repo):
    w, a = _argv(repo)
    with pytest.raises(DockerExecutionError, match="--rm"):
        validate_docker_argv([x for x in a if x != "--rm"], w)
    i = a.index("--pull")
    with pytest.raises(DockerExecutionError, match="pull never"):
        validate_docker_argv(a[:i] + a[i + 2 :], w)


# 9-12 failures ---------------------------------------------------------------------------------
def test_timeout_removes_container_and_fails(repo):
    fake = FakeDocker(timeout=True)
    res, ex = _run(repo, fake, timeout_s=5)
    assert res.status == STATUS_FAIL and ex.internal_records[0]["timed_out"] is True
    assert any(c[:3] == ["docker", "rm", "-f"] for c in fake.calls)
    assert res.qa["classification"]["fail_reasons"] == ["execution_failure"]


def test_nonzero_exit_fails(repo):
    res, ex = _run(repo, FakeDocker(exit_code=3, stderr="boom"))
    assert res.status == STATUS_FAIL and ex.internal_records[0]["exit_code"] == 3
    assert res.mask_path is None


def test_missing_output_mask_fails(repo):
    res, ex = _run(repo, FakeDocker(write_mask=False))
    assert res.status == STATUS_FAIL and ex.internal_records[0]["mask_present"] is False


def test_leftover_container_fails(repo):
    res, _ = _run(repo, FakeDocker(leftover=1))
    assert res.status == STATUS_FAIL


def test_engine_not_running_fails(repo):
    res, ex = _run(repo, FakeDocker(engine=False))
    assert res.status == STATUS_FAIL and ex.internal_records == []


def test_stdout_stderr_are_truncated(repo):
    res, ex = _run(repo, FakeDocker(stdout="a" * 10000, stderr="b" * 9000))
    rec = ex.internal_records[0]
    assert len(rec["stdout"]) < 4100 and rec["stdout"].endswith("[truncated 6000]")
    assert len(rec["stderr"]) < 4100
    assert _truncate("short") == "short"


# 13-17 audit / paths / injection ---------------------------------------------------------------
def test_host_paths_removed_from_public_audit(repo):
    res, ex = _run(repo, FakeDocker())
    folder = repo / "results/brain_extraction_pilot/_smoke/smoke-x-01/synthstrip"
    for name in ("audit.json", "qa.json", "run.log", "figure_info.json"):
        text = (folder / name).read_text(encoding="utf-8")
        assert str(repo) not in text and repo.as_posix() not in text
        assert not find_host_paths(text), name
    a = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    assert "<WORKDIR>/tool_input.nii.gz" in a["command"]
    assert "<WORKDIR>:/data" in a["execution"]["argv"]
    assert any(str(repo) in x for x in ex.internal_records[0]["argv"])  # internal keeps host paths


def test_internal_records_written_only_under_approved_root(repo):
    bad = repo / "elsewhere"
    with pytest.raises(DockerExecutionError, match="approved root"):
        DockerSynthStripExecutor(repo, run_fn=FakeDocker(), internal_dir=bad)
    d = repo / "results/brain_extraction_pilot/_internal/executions"
    _, ex = _run(repo, FakeDocker(), internal_dir=d)
    files = list(d.glob("*.json"))
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(files) == 1 and any(str(repo) in x for x in rec["argv"])


def test_user_name_is_removed(monkeypatch):
    monkeypatch.setenv("USERNAME", "zzneuroscanuser")
    s = sanitize_paths(["run by zzneuroscanuser", "/home/zzneuroscanuser/x", r"C:\Users\x\y"])
    assert s == ["run by [REDACTED]", "[REDACTED]", "[REDACTED]"]
    assert find_host_paths("owner zzneuroscanuser") and not find_host_paths("/data/x")
    with pytest.raises(Exception, match="host path"):
        guard_public({"x": "zzneuroscanuser"})


@pytest.mark.parametrize(
    "p", [r"C:\Projects\x", "D:/data/raw", r"E:\x", "C:/Users/a", r"\\srv\share"]
)
def test_drive_and_unc_paths_detected_and_redacted(p):
    assert find_host_paths(p)
    assert not find_host_paths(json.dumps(sanitize_paths([p])))


def test_original_ids_refused_in_public_record(repo):
    with pytest.raises(Exception, match="identifier"):
        guard_public({"execution": {"argv": ["sub-strokecase9993"]}})
    res, _ = _run(repo, FakeDocker())
    guard_public(res.audit)  # public audit passes the ID and host-path guard


def test_existing_results_not_overwritten(repo):
    _run(repo, FakeDocker())
    with pytest.raises(Exception, match="overwrite"):
        _run(repo, FakeDocker())


@pytest.mark.parametrize("bad", ["a;rm -rf x", "a|b", "$(x)", "a`b`", "a&b", "a>b", "a\nb"])
def test_shell_metacharacters_refused(bad):
    with pytest.raises(DockerExecutionError, match="metacharacter"):
        _check_args_safe(["docker", bad])


def test_injected_workdir_name_refused(repo):
    w = repo / "results/brain_extraction_pilot/_smoke/x" / "a;b.part"
    w.mkdir(parents=True)
    (w / INPUT_NAME).write_bytes(b"x")
    ex = DockerSynthStripExecutor(repo, run_fn=FakeDocker())
    with pytest.raises(DockerExecutionError, match="metacharacter"):
        ex(expected_tool_argv(w), w)


# 18-20 no MRI, no pull, no HD-BET --------------------------------------------------------------
def test_no_mri_access_no_pull_no_hdbet(repo, monkeypatch):
    opened = []
    real = nib.load
    monkeypatch.setattr(
        nib, "load", lambda p, *a, **k: (opened.append(Path(p).resolve()), real(p, *a, **k))[1]
    )
    fake = FakeDocker()
    _run(repo, fake)
    assert opened and all(repo.resolve() in p.parents for p in opened)  # synthetic files only
    assert not any(c[:2] == ["docker", "pull"] for c in fake.calls)
    run = [c for c in fake.calls if c[:2] == ["docker", "run"] and "--entrypoint" not in c][0]
    assert run[run.index("--pull") + 1] == "never"
    assert not any("hd-bet" in x or "hd_bet" in x for c in fake.calls for x in c)


def test_hdbet_command_refused_by_executor(repo):
    w = _workdir(repo)
    ex = DockerSynthStripExecutor(repo, run_fn=FakeDocker())
    with pytest.raises(DockerExecutionError, match="fixed policy"):
        ex(HDBetSpec().command(str(w / INPUT_NAME), str(w / "out.nii.gz")), w)


def test_model_hash_check(repo):
    ex = DockerSynthStripExecutor(repo, run_fn=FakeDocker())
    assert ex.verify_model_hash() == PILOT_POLICY.model_sha256


def test_only_executor_module_starts_processes():
    pkg = REPO / "ai/neuroscan_ai/brain_extraction"
    for f in pkg.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        mods += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        if f.name == "executors.py":
            assert "subprocess" in mods
            assert not {"urllib", "requests", "socket", "http", "torch", "HD_BET"} & {
                m.split(".")[0] for m in mods
            }
            src = f.read_text(encoding="utf-8")
            assert "shell=True" not in src and "os.system" not in src
        else:
            assert (
                "subprocess" not in mods and "neuroscan_ai.brain_extraction.executors" not in mods
            )
