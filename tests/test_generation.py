import torch


def _generation_function():
    try:
        from evaluation.generation import generate
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected evaluation.generation to exist: {error}") from error

    return generate


def _tiny_model():
    from model.transformer import MiniGPT

    torch.manual_seed(0)
    return MiniGPT(
        vocab_size=16,
        context_length=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
    )


def test_generate_appends_greedy_tokens_and_preserves_the_prompt() -> None:
    generate = _generation_function()
    model = _tiny_model()
    prompt = torch.tensor([[1, 2]])

    generated = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=0.0,
    )

    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :2], prompt)
    assert torch.all((generated >= 0) & (generated < 16))
    assert model.training


def test_top_one_sampling_matches_greedy_generation() -> None:
    generate = _generation_function()
    model = _tiny_model()
    prompt = torch.tensor([[1, 2]])

    greedy = generate(model, prompt, max_new_tokens=3, temperature=0.0)
    torch.manual_seed(123)
    top_one = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
    )

    assert torch.equal(greedy, top_one)


def test_generate_with_zero_new_tokens_returns_a_copy_of_the_prompt() -> None:
    generate = _generation_function()
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])

    generated = generate(model, prompt, max_new_tokens=0)

    assert torch.equal(generated, prompt)
    assert generated.data_ptr() != prompt.data_ptr()
