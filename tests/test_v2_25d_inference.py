"""v2 2.5D validation equals the 2D validation for a centre-slice model (synthetic, CPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from neuroscan_ai.v2 import baseline_25d_dataset as bd
from neuroscan_ai.v2 import baseline_25d_inference as bi
from neuroscan_ai.v2 import baseline_inference as ref2d  # comparison only
from neuroscan_ai.v2 import unet_2d  # comparison only
from neuroscan_ai.v2.cuda_smoke_25d import preserved_torch_state
from neuroscan_ai.v2.unet_25d import build_model
from tests.test_v2_25d_runner import make_world, open_world

CPU = torch.device("cpu")


class CentreOnly(nn.Module):
    """Feeds channels 2:4 (DWI_z, ADC_z) to a 2D model."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x[:, 2:4])


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1, *x.shape[-2:]), self.value)


@pytest.fixture(autouse=True)
def _restore():
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def val(tmp_path_factory):
    return open_world(make_world(tmp_path_factory.mktemp("inf25d")))[1]


def test_centre_wrapper_equals_2d_validation_exactly(val):
    net = unet_2d.build_model(3)
    wrap = CentreOnly(net)
    for c in val:
        a = bi.predict_model_grid(wrap, c, CPU, 2)
        b = ref2d.predict_model_grid(net, c, CPU, 2)
        assert a.tobytes() == b.tobytes()
    assert bi.validation_loss(wrap, val, CPU, 4) == ref2d.validation_loss(net, val, CPU, 4)
    ea, eb = bi.evaluate(wrap, val, CPU, 4), ref2d.evaluate(net, val, CPU, 4)
    for k in (
        "ungated_summary",
        "gated_summary",
        "ungated_vs_gated",
        "collapse",
        "per_case_ungated",
        "per_case_gated",
    ):
        assert json.dumps(ea[k]) == json.dumps(eb[k]), k
    for k, o in ea["outputs"].items():
        for name in ("probability", "ungated", "gated"):
            assert getattr(o, name).tobytes() == getattr(eb["outputs"][k], name).tobytes()


def test_prediction_of_slice_z_is_the_stack_centred_on_z(val):
    net = build_model(0).eval()
    c = val[0]
    prob = bi.predict_model_grid(net, c, CPU, 3)
    assert prob.shape == c.model_grid_shape
    h, w = c.model_grid_shape[:2]
    for z in range(c.model_grid_shape[2]):
        x = torch.from_numpy(bd.slice_item(c, z)["image"])[None]
        with torch.no_grad():
            p = torch.sigmoid(net(x))[0, 0, :h, :w].numpy()
        assert np.allclose(p, prob[:, :, z], atol=1e-6)


def test_every_input_channel_influences_the_prediction(val):
    net = build_model(0).eval()
    x = torch.from_numpy(bd.slice_item(val[0], 1)["image"])[None]
    with torch.no_grad():
        base = net(x)
        for c in range(6):
            y = x.clone()
            y[:, c, 4:12, 4:12] += 1.0
            assert not torch.equal(base, net(y))


def test_stop_conditions(val):
    with pytest.raises(bi.PredictionCollapseError):
        bi.evaluate(Constant(50.0), val, CPU, 4)
    with pytest.raises(bi.ProbabilityRangeError):
        bi.predict_model_grid(Constant(float("nan")), val[0], CPU, 4)
    with pytest.raises(bi.MissingValidationResultError):
        bi.evaluate(Constant(-50.0), [c for c in val if not c.original_gt.any()], CPU, 4)
    empty = bi.evaluate(Constant(-50.0), val, CPU, 4)
    assert empty["collapse"]["all_predictions_empty"] is True


def test_fixed_threshold_and_gated_subset(val):
    assert bi.THRESHOLD == 0.5
    ev = bi.evaluate(build_model(0), val, CPU, 4)
    for c in val:
        o = ev["outputs"][c.case_key]
        assert np.array_equal(o.ungated.astype(bool), o.probability >= np.float32(0.5))
        assert not (o.gated.astype(bool) & ~o.ungated.astype(bool)).any()
