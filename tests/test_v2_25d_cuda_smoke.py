"""Structure of the 2.5D CUDA memory smoke on a tiny CPU shape. The real smoke is never run here."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neuroscan_ai.v2 import cuda_smoke_25d as cs

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
TINY = (2, 6, 16, 16)
GIB = 1024**3


def _pass(**kw):
    p = {
        "steps": 3,
        "losses": [1.0, 0.9, 0.8],
        "gradient_norms": [1.0, 1.0, 1.0],
        "output_sha256": "a",
        "parameter_sha256": "b",
        "min_available_ram_bytes": 8 * GIB,
        "cuda_max_memory_reserved_bytes": 9 * GIB,
    }
    return {**p, **kw}


def test_formal_constants():
    assert cs.BATCH_SHAPE == (32, 6, 264, 264) and cs.MIN_STEPS == 3
    assert cs.MAX_RESERVED_GIB == 14.0 and cs.MIN_AVAILABLE_RAM_GIB == 4.0


def test_tiny_pass_is_deterministic_and_updates_parameters():
    with cs.preserved_torch_state():
        a, b = (cs.run_pass(CPU, shape=TINY) for _ in range(2))
        init = cs._sha_arrays(cs.build_model(cs.SMOKE_SEED).state_dict().values())
    assert a["losses"] == b["losses"] and a["parameter_sha256"] == b["parameter_sha256"]
    assert a["output_sha256"] == b["output_sha256"] and len(a["losses"]) == 3
    assert a["parameter_sha256"] != init and a["output_shape"] == [1, 1, 16, 16]
    with pytest.raises(ValueError):
        cs.run_pass(CPU, shape=TINY, steps=2)


def test_synthetic_batch_is_fp32_and_reproducible():
    a, b = cs.synthetic_batch(TINY, 42, 0), cs.synthetic_batch(TINY, 42, 0)
    assert all(v.dtype == torch.float32 for v in a.values())
    assert torch.equal(a["image"], b["image"]) and a["target"].any()
    assert not torch.equal(a["image"], cs.synthetic_batch(TINY, 42, 1)["image"])


@pytest.mark.parametrize(
    "kw,key",
    [
        ({"losses": [1.0, 0.9, 0.7]}, "same_losses"),
        ({"output_sha256": "x"}, "same_output"),
        ({"parameter_sha256": "x"}, "same_parameters"),
        ({"cuda_max_memory_reserved_bytes": 14 * GIB}, "max_memory_reserved_below_14_gib"),
        ({"cuda_max_memory_reserved_bytes": None}, "max_memory_reserved_below_14_gib"),
        ({"min_available_ram_bytes": 4 * GIB - 1}, "available_ram_at_least_4_gib"),
        ({"min_available_ram_bytes": None}, "available_ram_at_least_4_gib"),
        ({"losses": [1.0, 0.9, float("nan")]}, "all_values_finite"),
        ({"steps": 2}, "at_least_three_consecutive_batches"),
    ],
)
def test_every_condition_is_needed(kw, key):
    assert all(cs.evaluate_passes(_pass(), _pass()).values())
    assert cs.evaluate_passes(_pass(), _pass(**kw))[key] is False


def test_cpu_or_tiny_run_never_passes_and_folder_is_never_reused(tmp_path):
    out = tmp_path / "smoke"
    res = cs.run_smoke(out, device=CPU, context={}, shape=TINY)
    assert res["status"] == "failed" and res["checks"]["formal_shape_on_cuda"] is False
    assert [p.name for p in out.iterdir()] == [cs.FAILURE_NAME]
    with pytest.raises(FileExistsError):
        cs.run_smoke(out, device=CPU, context={}, shape=TINY)


@pytest.mark.parametrize(
    "err,oom",
    [(torch.cuda.OutOfMemoryError("synthetic"), True), (cs.bt.NonFiniteLossError("x"), False)],
)
def test_failure_recorded_once_without_retry_or_smaller_batch(tmp_path, err, oom):
    shapes = []

    def failing(device, *, shape):
        shapes.append(shape)
        raise err

    out = tmp_path / "f"
    res = cs.run_smoke(out, device=CPU, context={}, pass_fn=failing, shape=TINY)
    assert shapes == [TINY] and res["status"] == "failed" and res["cuda_oom"] is oom
    rec = json.loads((out / cs.FAILURE_NAME).read_text("utf-8"))
    assert rec["retried"] is False and rec["batch_size_changed"] is False
    assert rec["input_shape"] == list(TINY) and rec["completed_passes"] == 0
    assert "separate approval" in rec["next_step"]


def test_passing_record_structure(tmp_path):
    out = tmp_path / "p"
    cuda = torch.device("cuda")  # device name only: the pass function is synthetic
    res = cs.run_smoke(out, device=cuda, context={}, pass_fn=lambda d, *, shape: _pass())
    assert res["status"] == "passed" and (out / cs.RESULT_NAME).is_file()


def _state():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )


def test_torch_state_restored(tmp_path):
    s, t, n = _state(), torch.get_rng_state(), np.random.get_state()[1].copy()
    cs.run_smoke(tmp_path / "s", device=CPU, context={}, shape=TINY)
    assert _state() == s and torch.equal(torch.get_rng_state(), t)
    assert np.array_equal(np.random.get_state()[1], n)


def test_script_needs_approval_clean_tree_and_cuda(capsys, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "smoke25d", REPO / "scripts/smoke_v2_25d_cuda_memory.py"
    )
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    monkeypatch.setattr(sm, "run_smoke", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(sm, "OUT_DIR", tmp_path / "out")
    assert sm.main([]) == 2 and "approval" in capsys.readouterr().err
    monkeypatch.setattr(sm, "capture_source_control", lambda r: {"working_tree_clean": False})
    assert sm.main(["--approved"]) == 2 and "clean" in capsys.readouterr().err
    monkeypatch.setattr(sm, "capture_source_control", lambda r: {"working_tree_clean": True})
    monkeypatch.setattr(sm.torch.cuda, "is_available", lambda: False)
    assert sm.main(["--approved"]) == 2 and "cuda" in capsys.readouterr().err
    (tmp_path / "out").mkdir()
    assert sm.main(["--approved"]) == 2 and "exists" in capsys.readouterr().err


def test_smoke_code_opens_no_data():
    for f in ("ai/neuroscan_ai/v2/cuda_smoke_25d.py", "scripts/smoke_v2_25d_cuda_memory.py"):
        text = (REPO / f).read_text("utf-8")
        for word in ("nibabel", "isles_case_files", "baseline_cache", "_internal", "data/raw"):
            assert word not in text, (f, word)
