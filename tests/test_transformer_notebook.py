"""Regression tests for the self-contained educational Transformer notebook."""
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def ns():
    path = Path(__file__).resolve().parents[1] / "Simple Transformer Model" / "Transformer_model.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"][:-1]:  # Exclude the demonstration.
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    return namespace


def test_forward_and_backward(ns):
    model = ns["Transformer"](d_model=16, num_heads=4, num_encoder_layers=2,
                              d_ff=32, input_dim=25, max_len=6, dropout=0.0)
    out = model(torch.randint(0, 25, (3, 6)))
    assert out.shape == (3, 6, 25)
    out.square().mean().backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_encoder_layers_have_independent_parameters(ns):
    model = ns["Transformer"](d_model=8, num_heads=2, num_encoder_layers=2, input_dim=12)
    first, second = model.encoder.layers
    assert first is not second
    assert first.self_attn.query.weight.data_ptr() != second.self_attn.query.weight.data_ptr()


def test_attention_mask_and_fully_masked_rows(ns):
    q = torch.ones(1, 1, 2, 4)
    k = torch.ones(1, 1, 3, 4)
    v = torch.arange(12.0).reshape(1, 1, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    output, weights = ns["scaled_dot_product_attention"](q, k, v, mask)
    torch.testing.assert_close(output[0, 0, 0], v[0, 0, 0])
    assert torch.count_nonzero(weights[0, 0, 1]) == 0
    assert torch.count_nonzero(output[0, 0, 1]) == 0


def test_batch_first_positions_are_distinct(ns):
    pe = ns["PositionalEncoding"](7, max_len=4)
    x = pe(torch.zeros(2, 4, 7))
    torch.testing.assert_close(x[0], x[1])
    assert not torch.allclose(x[0, 0], x[0, 1])
    with pytest.raises(ValueError, match="max_len"):
        pe(torch.zeros(2, 5, 7))


def test_reject_invalid_heads_and_mask(ns):
    with pytest.raises(ValueError, match="divisible"):
        ns["MultiHeadAttention"](10, 3)
    q = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="broadcast"):
        ns["scaled_dot_product_attention"](q, q, q, torch.ones(2, 7, 7))


def test_notebook_demo_executes(ns):
    path = Path(__file__).resolve().parents[1] / "Simple Transformer Model" / "Transformer_model.ipynb"
    cell = json.loads(path.read_text(encoding="utf-8"))["cells"][-1]
    exec(compile("".join(cell["source"]), str(path), "exec"), dict(ns))
