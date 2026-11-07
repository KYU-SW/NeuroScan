"""Runners: v1 background baseline, SynthStrip / HD-BET specifications and mock runners.

No runner here starts a process, pulls an image or downloads weights. External tools need an
injected ``executor``; the default executor refuses. Mock runners stand in for a tool by writing a
synthetic mask file, so the full write -> read -> QA -> audit path is exercised without any tool.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel import orientations as ornt

from neuroscan_ai.brain_extraction import qa as qa_mod
from neuroscan_ai.brain_extraction.audit import (
    build_audit,
    guard_public,
    sanitize_paths,
    sha256_file,
    write_json_new,
)
from neuroscan_ai.brain_extraction.interfaces import (
    STATUS_FAIL,
    TOOL_HDBET,
    TOOL_SYNTHSTRIP,
    TOOL_V1,
    BrainExtractionError,
    RunRequest,
    RunResult,
)

MASK_NAME = "brain_mask.nii.gz"


class ExternalExecutionDisabled(RuntimeError):
    """Raised by the default executor: external tools are not enabled in this stage."""


def disabled_executor(argv: list[str], workdir: Path) -> None:
    raise ExternalExecutionDisabled("external tool execution is not enabled")


Executor = Callable[[list[str], Path], None]


# --------------------------------------------------------------------------------- tool specs
@dataclass(frozen=True)
class SynthStripSpec:
    """SynthStrip 1.8 command specification (not executed here)."""

    version: str = "1.8"
    container_image: str = "freesurfer/synthstrip:1.8"
    container_digest: str | None = None  # recorded at installation
    model_file: str = "synthstrip.1.pt"
    model_sha256: str | None = None  # recorded at installation
    border_mm: float = 1.0
    no_csf: bool = False  # CSF kept in the brain border (tool default)
    device: str = "cpu"  # "cpu" or "gpu"
    threads: int | None = None

    def command(self, image: str, mask: str) -> list[str]:
        argv = ["mri_synthstrip", "-i", image, "-m", mask, "-b", f"{self.border_mm:g}"]
        if self.no_csf:
            argv.append("--no-csf")
        if self.device == "gpu":
            argv.append("-g")
        if self.threads:
            argv += ["-t", str(self.threads)]
        return argv

    def mask_output(self, mask: str) -> str:
        return mask


@dataclass(frozen=True)
class HDBetSpec:
    """HD-BET 2.0.1 command specification (not executed here).

    HD-BET documents that inputs must match the MNI152 orientation (``fslreorient2std``); the
    runner therefore applies a lossless axis permutation / flip to ``standard_orientation`` before
    the tool and the exact inverse afterwards, and records both in the audit.
    """

    version: str = "2.0.1"
    weights_source: str = "zenodo record 14445620 (release_v1.5.0.zip)"
    weights_sha256: str | None = None  # recorded at installation
    tta: bool = True  # tool default (test-time mirroring)
    device: str = "cuda"  # "cuda", "cpu" or "mps"
    standard_orientation: str = "LAS"  # to be verified against fslreorient2std at installation

    def command(self, image: str, output: str) -> list[str]:
        argv = ["hd-bet", "-i", image, "-o", output, "-device", self.device, "--save_bet_mask"]
        if not self.tta:
            argv.append("--disable_tta")
        return argv

    def mask_output(self, output: str) -> str:
        if not output.endswith(".nii.gz"):
            raise BrainExtractionError("HD-BET output must end with .nii.gz")
        return output[: -len(".nii.gz")] + "_bet.nii.gz"


# ----------------------------------------------------------------------------- orientation
def to_orientation(img: nib.Nifti1Image, target: str) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Lossless axis permutation / flip so that the array axes follow ``target`` (no resampling)."""
    src = ornt.io_orientation(img.affine)
    dst = ornt.axcodes2ornt(tuple(target))
    tr = ornt.ornt_transform(src, dst)
    data = ornt.apply_orientation(np.asanyarray(img.dataobj), tr)
    aff = img.affine @ ornt.inv_ornt_aff(tr, img.shape)
    out = nib.Nifti1Image(data, aff, img.header.copy())
    out.set_qform(aff, code=int(img.header["qform_code"]) or 1)
    out.set_sform(aff, code=int(img.header["sform_code"]) or 1)
    rec = {
        "from": "".join(nib.aff2axcodes(img.affine)),
        "to": target,
        "transform": tr.tolist(),
        "lossless": True,
    }
    return out, rec


def from_orientation(
    mask: nib.Nifti1Image, original: nib.Nifti1Image
) -> tuple[nib.Nifti1Image, dict[str, Any]]:
    """Exact inverse of ``to_orientation`` back onto the original TRACE grid."""
    src = ornt.io_orientation(mask.affine)
    dst = ornt.io_orientation(original.affine)
    tr = ornt.ornt_transform(src, dst)
    data = ornt.apply_orientation(np.asanyarray(mask.dataobj), tr)
    out = nib.Nifti1Image(data, mask.affine @ ornt.inv_ornt_aff(tr, mask.shape), mask.header.copy())
    out.set_qform(out.affine, code=int(mask.header["qform_code"]) or 1)
    out.set_sform(out.affine, code=int(mask.header["sform_code"]) or 1)
    rec = {
        "from": "".join(nib.aff2axcodes(mask.affine)),
        "to": "".join(nib.aff2axcodes(original.affine)),
        "transform": tr.tolist(),
        "lossless": True,
    }
    return out, rec


def _mask_like(trace: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image:
    """uint8 0/1 image on the TRACE grid (same affine and qform/sform codes)."""
    hdr = trace.header.copy()
    hdr.set_data_dtype(np.uint8)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), trace.affine, hdr)
    img.header.set_slope_inter(1.0, 0.0)
    img.set_qform(trace.get_qform(), code=int(trace.header["qform_code"]))
    img.set_sform(trace.get_sform(), code=int(trace.header["sform_code"]))
    return img


# --------------------------------------------------------------------------------- base runner
class Runner:
    tool: str = ""
    mock: bool = False

    def spec_dict(self) -> dict[str, Any]:
        return {}

    def produce(
        self, trace: nib.Nifti1Image, trace_path: Path, work: Path
    ) -> tuple[nib.Nifti1Image, list[str] | None, bool, list[dict[str, Any]]]:
        """Return (mask image as written by the tool, command, external_executed, transforms)."""
        raise NotImplementedError

    def run(
        self,
        req: RunRequest,
        *,
        baseline: np.ndarray | None = None,
        thresholds: qa_mod.ReviewThresholds | None = None,
        forbidden_ids: tuple[str, ...] = (),
    ) -> RunResult:
        if req.tool != self.tool:
            raise BrainExtractionError(f"request is for '{req.tool}', runner is '{self.tool}'")
        final = Path(req.out_dir) / req.alias / req.tool
        work = final.with_name(final.name + ".part")
        if final.exists() or work.exists():
            raise BrainExtractionError("output folder exists; refusing to overwrite")
        trace = nib.load(str(req.trace_path))
        if trace.ndim != 3:
            raise BrainExtractionError("TRACE input must be 3D (use the format adapter output)")
        gt = None
        if req.gt_path is not None:
            gimg = nib.load(str(req.gt_path))
            if tuple(gimg.shape) != tuple(trace.shape) or not np.allclose(
                gimg.affine, trace.affine, atol=qa_mod.AFFINE_ATOL_MM
            ):
                raise BrainExtractionError("ground truth is not on the TRACE grid")
            gt = np.asarray(gimg.dataobj) > 0
        inputs = {"trace": sha256_file(req.trace_path)}
        if req.gt_path is not None:
            inputs["gt"] = sha256_file(req.gt_path)
        work.mkdir(parents=True)
        log = [f"alias={req.alias} tool={req.tool} version={req.tool_version} mock={self.mock}"]
        mask_img, command, executed, transforms, error = None, None, False, [], None
        try:
            mask_img, command, executed, transforms = self.produce(trace, req.trace_path, work)
            log.append("tool step finished")
        except Exception as e:  # noqa: BLE001 - recorded as an execution failure
            error = type(e).__name__
            log.append(f"tool step failed: {error}")
        # public records carry container / <WORKDIR> paths only; host paths stay internal
        command = sanitize_paths(command, [work]) if command is not None else None
        execution = getattr(getattr(self, "executor", None), "last_public_record", None)
        qa = qa_mod.run_qa(trace, mask_img, gt, baseline, thresholds, execution_error=error)
        status = qa["classification"]["status"]
        outputs: dict[str, str] = {}
        mask_path = None
        if "structure" in qa:  # geometry and values valid: canonical uint8 copy on the TRACE grid
            data = np.asarray(mask_img.dataobj, dtype=np.float64) == 1
            nib.save(_mask_like(trace, data), str(work / MASK_NAME))
            outputs[MASK_NAME] = sha256_file(work / MASK_NAME)
            mask_path = final / MASK_NAME
        audit = build_audit(
            alias=req.alias,
            tool=req.tool,
            tool_version=req.tool_version,
            spec=self.spec_dict(),
            command=command,
            executed_external=executed,
            mock=self.mock,
            transforms=transforms,
            input_sha256=inputs,
            output_sha256=outputs,
            status=status,
            execution_error=error,
        )
        if execution is not None:
            audit["execution"] = execution
        log.append(f"status={status}")
        figure_info = _figure_info(trace, gt)
        try:
            for obj in (qa, audit, log, figure_info):
                guard_public(obj, forbidden_ids)
            write_json_new(work / "qa.json", qa)
            write_json_new(work / "audit.json", audit)
            write_json_new(work / "figure_info.json", figure_info)
            (work / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        os.replace(work, final)
        return RunResult(req.alias, req.tool, status, mask_path, qa, audit, log, figure_info)


def _figure_info(trace: nib.Nifti1Image, gt: np.ndarray | None) -> dict[str, Any]:
    """Slice indices for the QA figure (no voxel data)."""
    info: dict[str, Any] = {"axial_mid_slice": int(trace.shape[2] // 2), "shape": list(trace.shape)}
    if gt is not None and gt.any():
        per = gt.sum(axis=(0, 1))
        info["gt_max_axial_slice"] = int(np.argmax(per))
        c = np.argwhere(gt).mean(axis=0).round().astype(int)
        info["gt_centroid_voxel"] = c.tolist()
    return info


# ---------------------------------------------------------------------------------- runners
class V1BackgroundRunner(Runner):
    """v1 frozen policy ``skull_stripped_background`` (imported read-only), the baseline."""

    tool = TOOL_V1

    def spec_dict(self) -> dict[str, Any]:
        return {"policy": "skull_stripped_background", "source": "neuroscan v1 (frozen)"}

    def produce(self, trace, trace_path, work):
        from neuroscan_ai.training.brain_gating import compute_brain_mask

        mask = compute_brain_mask("skull_stripped_background", trace.get_fdata(dtype=np.float32))
        return _mask_like(trace, mask), None, False, []


class ExternalToolRunner(Runner):
    """Builds the tool command and hands it to ``executor`` (default: refuses)."""

    def __init__(self, spec: SynthStripSpec | HDBetSpec, executor: Executor = disabled_executor):
        self.spec = spec
        self.executor = executor

    def spec_dict(self) -> dict[str, Any]:
        return {"name": type(self.spec).__name__, **asdict(self.spec)}

    def _execute(self, argv: list[str], work: Path, mask_file: Path) -> nib.Nifti1Image:
        self.executor(argv, work)
        if not mask_file.is_file():
            raise FileNotFoundError("tool produced no mask")
        return nib.load(str(mask_file))


class SynthStripRunner(ExternalToolRunner):
    tool = TOOL_SYNTHSTRIP

    def produce(self, trace, trace_path, work):
        inp, out = work / "tool_input.nii.gz", work / "tool_mask.nii.gz"
        shutil.copyfile(trace_path, inp)
        argv = self.spec.command(str(inp), str(out))
        img = self._execute(argv, work, Path(self.spec.mask_output(str(out))))
        return img, argv, not self.mock, []


class HDBetRunner(ExternalToolRunner):
    tool = TOOL_HDBET

    def produce(self, trace, trace_path, work):
        std, rec_in = to_orientation(trace, self.spec.standard_orientation)
        inp, out = work / "tool_input.nii.gz", work / "tool_output.nii.gz"
        nib.save(std, str(inp))
        argv = self.spec.command(str(inp), str(out))
        tool_mask = self._execute(argv, work, Path(self.spec.mask_output(str(out))))
        back, rec_out = from_orientation(tool_mask, trace)
        return back, argv, not self.mock, [rec_in, rec_out]


# ------------------------------------------------------------------------------- mock runners
MaskFn = Callable[[nib.Nifti1Image], nib.Nifti1Image | np.ndarray]


def _mock_executor(mask_fn: MaskFn, input_name: str, output_of: Callable[[list[str]], str]):
    """Executor that reads the tool input and writes ``mask_fn(input)`` where the tool would."""

    def run(argv: list[str], workdir: Path) -> None:
        img = nib.load(str(workdir / input_name))
        res = mask_fn(img)
        out = res if isinstance(res, nib.Nifti1Image) else _mask_like(img, res)
        nib.save(out, output_of(argv))

    return run


class MockSynthStripRunner(SynthStripRunner):
    mock = True

    def __init__(self, mask_fn: MaskFn, spec: SynthStripSpec | None = None):
        spec = spec or SynthStripSpec()
        super().__init__(spec, _mock_executor(mask_fn, "tool_input.nii.gz", lambda a: a[4]))


class MockHDBetRunner(HDBetRunner):
    mock = True

    def __init__(self, mask_fn: MaskFn, spec: HDBetSpec | None = None):
        spec = spec or HDBetSpec()
        super().__init__(
            spec,
            _mock_executor(mask_fn, "tool_input.nii.gz", lambda a: spec.mask_output(a[4])),
        )


def make_runner(tool: str, **kwargs: Any) -> Runner:
    if tool == TOOL_V1:
        return V1BackgroundRunner()
    if tool == TOOL_SYNTHSTRIP:
        return SynthStripRunner(
            kwargs.get("spec") or SynthStripSpec(), kwargs.get("executor", disabled_executor)
        )
    if tool == TOOL_HDBET:
        return HDBetRunner(
            kwargs.get("spec") or HDBetSpec(), kwargs.get("executor", disabled_executor)
        )
    raise BrainExtractionError(f"unknown tool '{tool}'")


__all__ = [
    "MASK_NAME",
    "STATUS_FAIL",
    "ExternalExecutionDisabled",
    "HDBetRunner",
    "HDBetSpec",
    "MockHDBetRunner",
    "MockSynthStripRunner",
    "SynthStripRunner",
    "SynthStripSpec",
    "V1BackgroundRunner",
    "disabled_executor",
    "from_orientation",
    "make_runner",
    "to_orientation",
]
