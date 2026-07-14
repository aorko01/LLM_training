import torch 
from types import SimpleNamespace

from model.attention import Attention

def test_attention_small_example():
    config = SimpleNamespace(
        n_embd=2,
        n_head=1,
    )

    model = Attention(config)

    # Make Q, K, V projections identities
    with torch.no_grad():
        for layer in (model.Wq, model.Wk, model.Wv):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()

    X = torch.tensor([[
        [1.0, 0.0],
        [0.0, 1.0],
    ]])

    output = model(X)

    expected = torch.tensor([[[[
        0.6698, 0.3302],
        [0.3302, 0.6698],
    ]]])

    assert torch.allclose(output, expected, atol=1e-4)