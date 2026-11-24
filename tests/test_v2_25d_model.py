"""v2 2.5D model: shape, parameter count, refusal, masked loss (synthetic, CPU)."""

from __future__ import annotations

import pytest
import torch

from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2 import unet_2d as ref2d  # comparison only
from neuroscan_ai.v2 import unet_25d as m25
from neuroscan_ai.v2.cuda_smoke_25d import preserved_torch_state


@pytest.fixture(autouse=True)
def _restore():
    with preserved_torch_state():
        yield


def test_parameter_count_and_formula():
    assert m25.count_parameters(m25.build_model(0)) == m25.EXPECTED_PARAMETERS == 2_440_873
    assert bt.expected_parameter_count(m25.CHANNELS, 6) == 2_440_873
    assert bt.expected_parameter_count(m25.CHANNELS, 2) == ref2d.EXPECTED_PARAMETERS
    # recorded candidates
    assert bt.expected_parameter_count((32, 64, 128, 256), 6) == 1_929_281
    assert bt.expected_parameter_count((28, 56, 112, 224), 6) == 1_477_785
    assert ref2d.EXPECTED_PARAMETERS + (6 * 2 + 2) == 2_439_591


def test_wrong_parameter_count_refused(monkeypatch):
    with pytest.raises(m25.ModelRefusedError):
        m25.check_parameters(m25.BaselineUNet25D(in_channels=2))
    monkeypatch.setattr(m25, "EXPECTED_PARAMETERS", 1)
    with pytest.raises(m25.ModelRefusedError):
        m25.build_model(0)


def test_only_first_conv_shape_differs_from_2d():
    a = {k: tuple(v.shape) for k, v in m25.build_model(42).state_dict().items()}
    b = {k: tuple(v.shape) for k, v in ref2d.build_model(42).state_dict().items()}
    assert list(a) == list(b)
    assert [k for k in a if a[k] != b[k]] == ["enc.0.0.weight"]
    assert a["enc.0.0.weight"] == (36, 6, 3, 3) and b["enc.0.0.weight"] == (36, 2, 3, 3)


def test_same_seed_is_not_a_paired_initialisation():
    a, b = m25.build_model(42).state_dict(), ref2d.build_model(42).state_dict()
    assert not torch.equal(a["enc.1.0.weight"], b["enc.1.0.weight"])


def test_same_forward_as_2d_architecture():
    """With the same weights the output is the same: the architecture is the 2D one."""
    net2 = ref2d.build_model(1)
    net25 = m25.BaselineUNet25D(in_channels=2)
    net25.load_state_dict(net2.state_dict())
    x = torch.randn(2, 2, 16, 24)
    with torch.no_grad():
        assert torch.equal(net25(x), net2(x))


def test_output_is_the_centre_slice_and_multiple_of_8():
    net = m25.build_model(0)
    assert net(torch.zeros(2, 6, 16, 24)).shape == (2, 1, 16, 24)
    for bad in ((1, 6, 12, 16), (1, 6, 16, 20)):
        with pytest.raises(ValueError, match="multiples of 8"):
            net(torch.zeros(bad))
    for bad in ((1, 2, 16, 16), (6, 16, 16)):
        with pytest.raises(ValueError):
            net(torch.zeros(bad))


def test_seed_determinism_and_description():
    a, b, c = (m25.build_model(s).state_dict() for s in (7, 7, 8))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)
    d = m25.describe(m25.build_model(7), 7)
    assert d["input_channels"] == ["dwi_z-1", "adc_z-1", "dwi_z", "adc_z", "dwi_z+1", "adc_z+1"]
    assert d["in_channels"] == 6 and d["out_channels"] == 1 and d["pretrained_weights"] is None


def test_masked_loss_gives_zero_gradient_in_padding():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 2:5, 2:5] = 1
    valid = torch.ones(2, 1, 16, 16)
    valid[:, :, 12:, :] = 0
    valid[:, :, :, 10:] = 0
    loss, _, _ = bt.combined_loss(logits, target, valid)
    loss.backward()
    assert torch.all(logits.grad[valid == 0] == 0) and torch.any(logits.grad[valid == 1] != 0)


def test_all_parameters_and_input_channels_get_finite_gradients():
    net = m25.build_model(0)
    y = torch.zeros(2, 1, 16, 16)
    y[:, :, 4:8, 4:8] = 1
    loss, _, _ = bt.combined_loss(net(torch.randn(2, 6, 16, 16)), y, torch.ones_like(y))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    g = net.enc[0][0].weight.grad
    assert all(bool((g[:, c] != 0).any()) for c in range(6))
