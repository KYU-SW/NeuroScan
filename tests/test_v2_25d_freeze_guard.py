"""Freeze guard: the 2D baseline files and the shared modules are unchanged (LF SHA-256 at commit
92bbffa), the 2.5D production code imports only approved modules and changes no shared state.
Later approved changes of a pinned file must update this pin explicitly."""

from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroscan_ai.v2 import baseline_25d_dataset as bd
from neuroscan_ai.v2 import cuda_smoke_25d as cs
from neuroscan_ai.v2 import unet_2d  # regression check only
from neuroscan_ai.v2 import unet_25d as m25

REPO = Path(__file__).resolve().parents[1]
FROZEN_2D = {
    "ai/neuroscan_ai/v2/unet_2d.py": (
        "06c0f35c3c2b4df2cea3aed1f1e34ea37b5d477b15de0ca300285a385fe58f3e"
    ),
    "ai/neuroscan_ai/v2/baseline_dataset.py": (
        "a9584aba06882b3b63b54d16539d9f365e7aec091a1ccdda398eb99cd46f065b"
    ),
    "ai/neuroscan_ai/v2/baseline_inference.py": (
        "a17210eb1adea8fd06e8806abcb404362f03cc1b1b8515f4de739f05e4ed9060"
    ),
    "ai/neuroscan_ai/v2/baseline_runner.py": (
        "d26780380f4e1583f2d27059b4d8ab9777d38df2d1b389b5e94ab801f1672135"
    ),
    "scripts/run_v2_2d_baseline.py": (
        "087f05b3c4f372c09e12307d221c67a19c0c7990f9ec9e09cc5cbffdb857e899"
    ),
    "configs/v2/2d_baseline.yaml": (
        "4531eab55b5f109a7c7f27a3ae777520d62bbb5424f6625285922c88e9760086"
    ),
}
SHARED = {
    "ai/neuroscan_ai/v2/baseline_training.py": (
        "6171b18a5cb4892407a20a675af72a9ad17a45eca460ad9f12a971a024cb3bbe"
    ),
    "ai/neuroscan_ai/v2/baseline_slices.py": (
        "a431cb009a23418fca878c159065100641c1f05de990a6aac8a031664a6dae2d"
    ),
    "ai/neuroscan_ai/v2/baseline_preprocess.py": (
        "72f48d063452aebb7238b6a49c13353fc51fbe565a9fbc521d936872bb0cab40"
    ),
    "ai/neuroscan_ai/v2/baseline_cache.py": (
        "7b39beb6a521de86c791787bf2b136f765155be446e040ef172bfe63e662c0e3"
    ),
    "ai/neuroscan_ai/v2/geometry.py": (
        "6fee3e7e3ce320fd820105ff26ffb66951ac0ad03db692dbb5f3c6391d7d7730"
    ),
    "ai/neuroscan_ai/v2/safety_metrics.py": (
        "9d8a4aa8dbaaa7a5f3b518109ff73cd4ddbff5cf7a8a69b184ceb21824c00eef"
    ),
    "ai/neuroscan_ai/v2/baseline_selection.py": (
        "1994a5a0f15e1da6bd71919e5bd9b0a728e95251f29cb8a26a523a22cbc52124"
    ),
    "ai/neuroscan_ai/v2/baseline_outputs.py": (
        "66d3d3001d5f13559d8205af38ec92e89743e472d04256f029e6fc5dbce72ae9"
    ),
    "ai/neuroscan_ai/v2/baseline_progress.py": (
        "127db30d93f1a363745b9c08c43c184811446c8b76b248e921cbf06263e852c7"
    ),
    "ai/neuroscan_ai/v2/baseline_resume.py": (
        "81e7331a0c72c5fd39cd3856ae6aa2bcd602e2455cc76a67e7a8219f24b9c08e"
    ),
    "ai/neuroscan_ai/v2/resource_monitor.py": (
        "e8bed7d4e1db2054b4eb32603f02b71c8c0e27e26ee757b706daffedf5421bf7"
    ),
    "ai/neuroscan_ai/v2/outputs.py": (
        "d0a53b66e452de10ce1ae12410f20873a2484224901e9c5bebfd0298bdcd9282"
    ),
    "ai/neuroscan_ai/v2/dataset.py": (
        "01f0bc4d8b77e2689c34d0d4f0332a84648d89a3c253f850605d74fc3d7fdb94"
    ),
    "ai/neuroscan_ai/run_control/__init__.py": (
        "6d1e118017c10d80d045a5db1975728c86575817ca95fa19a0ac19e867729f7d"
    ),
    "ai/neuroscan_ai/run_control/bundle.py": (
        "e7f376edf00a1f3987e983efd9ba3206485fa391542f93b3756161c2c624a46b"
    ),
    "ai/neuroscan_ai/run_control/gate.py": (
        "22b6a8fcabfd77eb319a0a5035373bbffd5c850c0855252b92e96ac0d0d6e3d0"
    ),
    "ai/neuroscan_ai/run_control/lifecycle.py": (
        "12655ae8aa6d0c9f9570e91146983741956e7f7c5d0546d6988d13f35914a882"
    ),
    "ai/neuroscan_ai/run_control/outputs.py": (
        "74f6b1f4d2a860c79a3fc8809c54c304456bbac47977c0b582246d40cec3b947"
    ),
    "ai/neuroscan_ai/run_control/plan.py": (
        "5472f457c3560ebc27a19ee6761f14a87172d6fb3791e16eb6b8d4f006f4d990"
    ),
    "ai/neuroscan_ai/run_control/runner.py": (
        "afdaf77eea57f8fde6a006a7a8fcdd541d7d62fe21a0af0251241a88b7526df2"
    ),
    "ai/neuroscan_ai/reproducibility/__init__.py": (
        "3993f0eb70a10cb4be3c92ad8af2b94dfd27dfd90593d632082d20237ff9cb5f"
    ),
    "ai/neuroscan_ai/reproducibility/environment.py": (
        "c94da443f1e47e1abf44e4acc197c3bc01ad289e05532dadff077525dbdea9f4"
    ),
    "ai/neuroscan_ai/reproducibility/hashing.py": (
        "210e806c50dbbd284f3dbfdcc981fc779d7beabd34ab55b5e13290d66d94a600"
    ),
    "ai/neuroscan_ai/reproducibility/manifest.py": (
        "d82767c7d5b316c9b8e39e4c722d11e4ffb0a4ede68fa3432e574dcc964f8305"
    ),
    "ai/neuroscan_ai/reproducibility/validation.py": (
        "4a59238c0afa93f47b48c32720d7df07278d2dde8edc3852cd4c3c97536f8239"
    ),
}
PRODUCTION = (
    "ai/neuroscan_ai/v2/unet_25d.py",
    "ai/neuroscan_ai/v2/baseline_25d_dataset.py",
    "ai/neuroscan_ai/v2/baseline_25d_inference.py",
    "ai/neuroscan_ai/v2/baseline_25d_runner.py",
    "ai/neuroscan_ai/v2/cuda_smoke_25d.py",
    "scripts/run_v2_25d_baseline.py",
    "scripts/prepare_v2_25d_cache.py",
    "scripts/smoke_v2_25d_cuda_memory.py",
)
FORBIDDEN = {
    "unet_2d",
    "baseline_dataset",
    "baseline_inference",
    "baseline_runner",
    "run_v2_2d_baseline",
}
ALLOWED_V2 = {Path(p).stem for p in SHARED if "/v2/" in p} | {
    "unet_25d",
    "baseline_25d_dataset",
    "baseline_25d_inference",
    "baseline_25d_runner",
    "cuda_smoke_25d",
}


def _sha(path: str) -> str:
    return hashlib.sha256((REPO / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("path", list(FROZEN_2D))
def test_frozen_2d_files_unchanged(path):
    assert _sha(path) == FROZEN_2D[path]


@pytest.mark.parametrize("path", list(SHARED))
def test_shared_modules_unchanged(path):
    assert _sha(path) == SHARED[path]


def _imports(text: str) -> set[str]:
    """Dotted names imported by a source, incl. ``from package import module`` and dynamic ones."""
    out: set[str] = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "relative import"
            out |= {f"{n.module}.{a.name}" for a in n.names} | {str(n.module)}
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            assert name not in ("import_module", "__import__"), "dynamic import"
    return out


@pytest.mark.parametrize("path", PRODUCTION)
def test_production_imports(path):
    names = _imports((REPO / path).read_text("utf-8"))
    parts = {p for n in names for p in n.split(".")}
    assert not parts & FORBIDDEN
    for n in names:
        if not n.startswith("neuroscan_ai"):
            assert n.split(".")[0] != "scripts"
            continue
        seg = n.split(".")
        if len(seg) == 1 or n == "neuroscan_ai.v2":
            continue
        if seg[1] == "v2":
            assert seg[2] in ALLOWED_V2, n
        else:
            assert seg[1] in ("run_control", "reproducibility"), n


def test_import_check_detects_forbidden_forms():
    for text in (
        "from neuroscan_ai.v2 import unet_2d",
        "from neuroscan_ai.v2.baseline_runner import prepare_config",
        "import neuroscan_ai.v2.baseline_dataset as x",
    ):
        assert {p for n in _imports(text) for p in n.split(".")} & FORBIDDEN
    with pytest.raises(AssertionError):
        _imports("import importlib\nimportlib.import_module('x')")


@pytest.mark.parametrize("path", PRODUCTION)
def test_no_assignment_to_imported_module_attributes(path):
    """No monkey patch: no attribute of an imported project module is assigned or deleted."""
    tree = ast.parse((REPO / path).read_text("utf-8"))
    mods = {"bc", "bt", "bp", "brs", "rm", "bd", "bi"}
    for n in ast.walk(tree):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        elif isinstance(n, ast.Delete):
            targets = n.targets
        for t in targets:
            while isinstance(t, (ast.Attribute, ast.Subscript)):
                t = t.value
                assert not (isinstance(t, ast.Name) and t.id in mods), ast.dump(n)
        if isinstance(n, ast.Call):
            assert getattr(n.func, "id", "") not in ("setattr", "delattr")


def _constants() -> dict[str, str]:
    out = {}
    for p in (*SHARED, *FROZEN_2D):
        if not p.startswith("ai/"):
            continue
        name = p.removeprefix("ai/").removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        mod = importlib.import_module(name)
        for k, v in vars(mod).items():
            if k.startswith("__"):
                continue
            plain = isinstance(v, (int, float, str, bool, tuple, list, dict, frozenset, type(None)))
            out[f"{name}.{k}"] = repr(v) if plain else f"id:{id(v)}"
    return out


def test_shared_state_and_2d_results_unchanged_by_25d_code():
    with cs.preserved_torch_state():
        x = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 2, 16, 24))).float()

        def two_d():
            with torch.no_grad():
                return unet_2d.build_model(5)(x).numpy().tobytes()

        consts, ref = _constants(), two_d()
        cs.run_pass(torch.device("cpu"), shape=(2, 6, 16, 16))
        v = np.random.default_rng(1).standard_normal((2, 16, 16, 3)).astype(np.float32)
        net = m25.build_model(1).eval()
        with torch.no_grad():
            net(torch.from_numpy(bd.stack_slices(v, 0))[None])
        assert two_d() == ref and _constants() == consts
