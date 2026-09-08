import pytest
import torch


def _transformer_module():
    try:
        from model.transformer import (
            MiniGPT,
            MultiHeadSelfAttention,
            TransformerBlock,
            causal_mask,
            scaled_dot_product_attention,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected model.transformer to exist: {error}", pytrace=False)

    return (
        MiniGPT,
        MultiHeadSelfAttention,
        TransformerBlock,
        causal_mask,
        scaled_dot_product_attention,
    )


def test_causal_mask_allows_current_and_previous_positions_only() -> None:
    _, _, _, causal_mask, _ = _transformer_module()

    mask = causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert torch.equal(mask, expected)


def test_scaled_dot_product_attention_preserves_shape() -> None:
    _, _, _, _, scaled_dot_product_attention = _transformer_module()
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 4, 5, 8)
    value = torch.randn(2, 4, 5, 8)

    output = scaled_dot_product_attention(query, key, value)

    assert output.shape == query.shape


def test_attention_respects_the_causal_mask() -> None:
    _, _, _, causal_mask, scaled_dot_product_attention = _transformer_module()
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros(1, 1, 3, 1)
    value = torch.tensor([[[[1.0], [2.0], [4.0]]]])

    output = scaled_dot_product_attention(query, key, value, mask=causal_mask(3))
    expected = torch.tensor([[[[1.0], [1.5], [7.0 / 3.0]]]])

    assert torch.allclose(output, expected)


def test_multi_head_self_attention_preserves_shape() -> None:
    _, MultiHeadSelfAttention, _, _, _ = _transformer_module()
    attention = MultiHeadSelfAttention(d_model=16, n_heads=4)
    inputs = torch.randn(2, 5, 16)

    output = attention(inputs)

    assert output.shape == inputs.shape


def test_transformer_block_has_an_identity_residual_path() -> None:
    _, _, TransformerBlock, _, _ = _transformer_module()
    block = TransformerBlock(d_model=16, n_heads=4, d_ff=32)
    inputs = torch.randn(2, 5, 16)

    with torch.no_grad():
        for parameter in block.attention.parameters():
            parameter.zero_()
        for parameter in block.feed_forward.parameters():
            parameter.zero_()

    output = block(inputs)

    assert torch.allclose(output, inputs)


def test_mini_gpt_returns_logits_with_expected_shape() -> None:
    MiniGPT, _, _, _, _ = _transformer_module()
    model = MiniGPT(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
    )
    input_ids = torch.randint(0, 32, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, 32)


def test_mini_gpt_cannot_use_future_tokens_for_previous_logits() -> None:
    MiniGPT, _, _, _, _ = _transformer_module()
    torch.manual_seed(0)
    model = MiniGPT(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
    )
    model.eval()
    first_sequence = torch.tensor([[1, 2, 3, 4]])
    second_sequence = torch.tensor([[1, 9, 8, 7]])

    with torch.no_grad():
        first_logits = model(first_sequence)
        second_logits = model(second_sequence)

    assert torch.allclose(first_logits[:, 0, :], second_logits[:, 0, :], atol=1e-6)
