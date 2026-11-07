"""SynthStrip Docker executor for the brain-extraction pilot (the only module that runs a process).

Policy ``bm_synthstrip_1_8_cpu_pilot`` is fixed in code and cannot be changed per case:

* image pinned by digest (``freesurfer/synthstrip@sha256:ebbc1772...``), never by tag, and it must
  already be present locally (``--pull never``; this module never pulls an image)
* ``--rm --network none``, CPU only (no ``--gpus``), 4 threads, border 1 mm, CSF kept
* exactly one bind mount: the runner's isolated work folder (``<alias>/<tool>.part``) -> ``/data``;
  the folder must lie under an approved output root and contain only the tool input
* no privileged mode, host network, extra mounts, Docker socket, environment variables or shell
* timeout with forced container removal, exit code / stdout / stderr / duration recorded, output
  mask checked, leftover containers checked

Records: ``internal_records`` keep host paths (local only, never public); ``last_public_record``
is path-sanitised (``<WORKDIR>``) and goes into the runner audit.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neuroscan_ai.brain_extraction.audit import sanitize_paths
from neuroscan_ai.brain_extraction.runners import SynthStripSpec


@dataclass(frozen=True)
class SynthStripPolicy:
    name: str = "bm_synthstrip_1_8_cpu_pilot"
    repository: str = "freesurfer/synthstrip"
    version: str = "1.8"
    digest: str = "sha256:ebbc177221194371f16362513ace68312a22922bb581bdfa618ac7ff9c1d2c06"
    model_file: str = "synthstrip.1.pt"
    model_sha256: str = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"
    model_path_in_image: str = "/freesurfer/models/synthstrip.1.pt"
    device: str = "cpu"
    threads: int = 4
    border_mm: float = 1.0
    no_csf: bool = False  # CSF kept in the brain border
    network: str = "none"

    @property
    def image(self) -> str:
        return f"{self.repository}@{self.digest}"


PILOT_POLICY = SynthStripPolicy()
CONTAINER_DIR = "/data"
INPUT_NAME, MASK_NAME = "tool_input.nii.gz", "tool_mask.nii.gz"
APPROVED_ROOTS = (
    "results/brain_extraction_pilot",
    "data/interim/soop_brain_masks",
    "data/interim/isles_brain_masks",
)
# Commands are always argv lists with shell=False; these characters are refused anyway as a second
# guard against injection (Go-template braces of ``--format`` are allowed).
_SHELL_META = set(";|&$`<>\n\r\"'")
_FORBIDDEN_FLAGS = {
    "--privileged", "--gpus", "--mount", "-e", "--env", "--env-file", "--pid", "--ipc", "--uts",
    "--cap-add", "--device", "--security-opt", "--volumes-from", "--userns", "--cgroupns",
    "--runtime", "--add-host", "-p", "--publish", "-P", "--publish-all",
}  # fmt: skip
MAX_OUTPUT_CHARS = 4000


class DockerExecutionError(RuntimeError):
    """Container could not be run as specified (message never contains a host path)."""


def pilot_spec(policy: SynthStripPolicy = PILOT_POLICY) -> SynthStripSpec:
    """The only SynthStrip spec accepted by the executor."""
    return SynthStripSpec(
        version=policy.version,
        container_image=f"{policy.repository}:{policy.version}",
        container_digest=policy.digest,
        model_file=policy.model_file,
        model_sha256=policy.model_sha256,
        border_mm=policy.border_mm,
        no_csf=policy.no_csf,
        device=policy.device,
        threads=policy.threads,
    )


def expected_tool_argv(workdir: Path, policy: SynthStripPolicy = PILOT_POLICY) -> list[str]:
    return pilot_spec(policy).command(str(workdir / INPUT_NAME), str(workdir / MASK_NAME))


def _check_args_safe(args: Sequence[str]) -> None:
    for a in args:
        if not isinstance(a, str) or any(ch in _SHELL_META for ch in a):
            raise DockerExecutionError("argument contains a shell metacharacter")


def _approved_workdir(workdir: Path, repo_root: Path, roots: Sequence[str]) -> Path:
    w = workdir.resolve()
    repo = repo_root.resolve()
    if w == repo or w in repo.parents or (w / ".git").exists():
        raise DockerExecutionError("refusing to mount the repository or an ancestor")
    allowed = [(repo / r).resolve() for r in roots]
    if not any(a in w.parents for a in allowed):
        raise DockerExecutionError("work folder is not under an approved output root")
    if not w.name.endswith(".part"):
        raise DockerExecutionError("work folder must be the runner's isolated .part folder")
    contents = sorted(p.name for p in w.iterdir())
    if contents != [INPUT_NAME]:
        raise DockerExecutionError("work folder must contain only the tool input")
    return w


def build_docker_argv(
    workdir: Path, container_name: str, policy: SynthStripPolicy = PILOT_POLICY
) -> list[str]:
    host = str(workdir.resolve())
    return [
        "docker", "run", "--rm", "--pull", "never", "--network", policy.network,
        "--name", container_name, "-v", f"{host}:{CONTAINER_DIR}", policy.image,
        "-i", f"{CONTAINER_DIR}/{INPUT_NAME}", "-m", f"{CONTAINER_DIR}/{MASK_NAME}",
        "-b", f"{policy.border_mm:g}", "-t", str(policy.threads),
    ]  # fmt: skip


def validate_docker_argv(
    argv: Sequence[str], workdir: Path, policy: SynthStripPolicy = PILOT_POLICY
) -> None:
    """Independent check of the final command against the fixed policy."""
    a = list(argv)
    if a[:2] != ["docker", "run"]:
        raise DockerExecutionError("not a docker run command")
    if any(x in _FORBIDDEN_FLAGS or x.startswith(("--gpus", "--privileged", "--mount")) for x in a):
        raise DockerExecutionError("forbidden docker option")
    if "--rm" not in a:
        raise DockerExecutionError("--rm is required")
    if a.count("--network") != 1 or a[a.index("--network") + 1] != "none":
        raise DockerExecutionError("--network none is required")
    if a.count("--pull") != 1 or a[a.index("--pull") + 1] != "never":
        raise DockerExecutionError("--pull never is required")
    vols = [i for i, x in enumerate(a) if x in ("-v", "--volume")]
    if len(vols) != 1:
        raise DockerExecutionError("exactly one mount is allowed")
    spec = a[vols[0] + 1]
    host, _, cont = spec.rpartition(":")
    if cont != CONTAINER_DIR or Path(host).resolve() != workdir.resolve():
        raise DockerExecutionError("mount must be the work folder on /data")
    if "docker.sock" in spec:
        raise DockerExecutionError("docker socket mount is forbidden")
    images = [x for x in a if x.startswith(policy.repository)]
    if images != [policy.image]:
        raise DockerExecutionError("image must be the approved digest (tag-only is refused)")
    tail = a[a.index(policy.image) + 1 :]
    expected = build_docker_argv(workdir, "x", policy)
    if tail != expected[expected.index(policy.image) + 1 :]:
        raise DockerExecutionError("tool arguments differ from the fixed policy")


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"...[truncated {len(text) - limit}]"


RunFn = Callable[..., subprocess.CompletedProcess]


class DockerSynthStripExecutor:
    """Executor for ``SynthStripRunner``: ``executor(tool_argv, workdir)``."""

    def __init__(
        self,
        repo_root: Path,
        *,
        policy: SynthStripPolicy = PILOT_POLICY,
        timeout_s: float = 600.0,
        approved_roots: Sequence[str] = APPROVED_ROOTS,
        run_fn: RunFn = subprocess.run,
        internal_dir: Path | None = None,
    ):
        if policy != PILOT_POLICY:
            raise DockerExecutionError("only the approved pilot policy can be executed")
        self.repo_root = Path(repo_root)
        self.policy = policy
        self.timeout_s = timeout_s
        self.approved_roots = tuple(approved_roots)
        self._run = run_fn
        self.internal_dir = Path(internal_dir) if internal_dir is not None else None
        if self.internal_dir is not None:
            d = self.internal_dir.resolve()
            allowed = [(self.repo_root.resolve() / r) for r in self.approved_roots]
            if not any(a == d or a in d.parents for a in allowed):
                raise DockerExecutionError("internal record folder is not under an approved root")
        self.internal_records: list[dict[str, Any]] = []
        self.last_public_record: dict[str, Any] | None = None

    # -- docker helpers (argv lists only, never a shell) ------------------------------------------
    def _docker(self, args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
        _check_args_safe(args)
        return self._run(args, capture_output=True, text=True, timeout=timeout, shell=False)

    def check_engine(self) -> str:
        p = self._docker(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
        if p.returncode != 0 or not p.stdout.strip():
            raise DockerExecutionError("docker engine is not running")
        return p.stdout.strip()

    def check_image(self) -> str:
        """The approved image must already be local (this module never pulls)."""
        p = self._docker(["docker", "image", "inspect", self.policy.image, "--format", "{{.Id}}"])
        if p.returncode != 0 or p.stdout.strip() != self.policy.digest:
            raise DockerExecutionError("approved image is not present locally")
        return p.stdout.strip()

    def verify_model_hash(self) -> str:
        p = self._docker([
            "docker", "run", "--rm", "--pull", "never", "--network", "none",
            "--entrypoint", "sha256sum", self.policy.image, self.policy.model_path_in_image,
        ], timeout=120)  # fmt: skip
        got = p.stdout.split()[0] if p.returncode == 0 and p.stdout.split() else ""
        if got != self.policy.model_sha256:
            raise DockerExecutionError("model hash differs from the approved policy")
        return got

    def leftover_containers(self, name: str) -> int:
        p = self._docker(["docker", "ps", "-a", "-q", "--filter", f"name={name}"])
        return len(p.stdout.split()) if p.returncode == 0 else -1

    # -- main entry ------------------------------------------------------------------------------
    def __call__(self, tool_argv: list[str], workdir: Path) -> None:
        self.last_public_record = None  # never carry a previous case's record into this audit
        workdir = Path(workdir)
        _check_args_safe([workdir.name])
        if list(tool_argv) != expected_tool_argv(workdir, self.policy):
            raise DockerExecutionError("tool arguments differ from the fixed policy")
        w = _approved_workdir(workdir, self.repo_root, self.approved_roots)
        engine = self.check_engine()
        self.check_image()
        name = f"neuroscan-bet-{uuid.uuid4().hex[:12]}"
        argv = build_docker_argv(w, name, self.policy)
        validate_docker_argv(argv, w, self.policy)
        _check_args_safe([x for x in argv if not x.startswith(str(w))])
        rec: dict[str, Any] = {
            "policy": asdict(self.policy),
            "engine_version": engine,
            "container_name": name,
            "argv": argv,
            "timeout_s": self.timeout_s,
        }
        t0 = time.monotonic()
        try:
            p = self._run(argv, capture_output=True, text=True, timeout=self.timeout_s, shell=False)
            rec.update(exit_code=p.returncode, stdout=_truncate(p.stdout or ""),
                       stderr=_truncate(p.stderr or ""), timed_out=False)  # fmt: skip
        except subprocess.TimeoutExpired as e:
            self._docker(["docker", "rm", "-f", name], timeout=60)
            rec.update(
                exit_code=None,
                timed_out=True,
                stdout=_truncate(str(e.stdout or "")),
                stderr=_truncate(str(e.stderr or "")),
            )
        rec["seconds"] = round(time.monotonic() - t0, 3)
        rec["leftover_containers"] = self.leftover_containers(name)
        rec["mask_present"] = (w / MASK_NAME).is_file()
        self.internal_records.append(rec)
        if self.internal_dir is not None:  # host paths allowed here: local, git-ignored, not public
            self.internal_dir.mkdir(parents=True, exist_ok=True)
            out = self.internal_dir / f"{name}.json"
            if out.exists():
                raise DockerExecutionError("internal record exists")
            out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        public = sanitize_paths(rec, [w, workdir])
        self.last_public_record = public
        if rec["timed_out"]:
            raise DockerExecutionError("synthstrip timed out")
        if rec["exit_code"] != 0:
            raise DockerExecutionError(f"synthstrip exited with code {rec['exit_code']}")
        if rec["leftover_containers"] != 0:
            raise DockerExecutionError("container was not removed")
        if not rec["mask_present"]:
            raise DockerExecutionError("synthstrip produced no mask")


__all__ = [
    "APPROVED_ROOTS",
    "PILOT_POLICY",
    "DockerExecutionError",
    "DockerSynthStripExecutor",
    "SynthStripPolicy",
    "build_docker_argv",
    "expected_tool_argv",
    "pilot_spec",
    "validate_docker_argv",
]
