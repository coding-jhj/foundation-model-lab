from __future__ import annotations

import torch
from torch import nn

from model.math_ops import softmax


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressively append tokens using greedy or temperature sampling."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    context_length = getattr(model, "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("model must expose a positive integer context_length")

    generated = input_ids.clone()
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = generated[:, -context_length:]
            logits = model(context)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                if top_k is not None:
                    number_of_candidates = min(top_k, next_token_logits.shape[-1])
                    top_values = torch.topk(
                        next_token_logits,
                        number_of_candidates,
                        dim=-1,
                    ).values
                    cutoff = top_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < cutoff,
                        torch.finfo(next_token_logits.dtype).min,
                    )

                probabilities = softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
    finally:
        model.train(was_training)

    return generated
