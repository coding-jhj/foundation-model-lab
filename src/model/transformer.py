from __future__ import annotations

import math

import torch
from torch import nn

from model.math_ops import softmax


def causal_mask(
    sequence_length: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean mask that blocks attention to future positions."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    return torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=device,
        )
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention for [batch, heads, sequence, features]."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have four dimensions")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value must share batch and head dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same feature dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")

    head_dimension = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dimension)

    if mask is not None:
        scores = scores.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(scores.dtype).min)

    attention_weights = softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dimension = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to [batch, sequence, d_model] inputs."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("inputs must have shape [batch, sequence, d_model]")

        batch_size, sequence_length, _ = inputs.shape

        query = self._split_heads(self.query_projection(inputs), batch_size, sequence_length)
        key = self._split_heads(self.key_projection(inputs), batch_size, sequence_length)
        value = self._split_heads(self.value_projection(inputs), batch_size, sequence_length)

        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=causal_mask(sequence_length, device=inputs.device),
        )
        merged = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.d_model,
        )
        return self.output_projection(merged)

    def _split_heads(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        return tensor.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dimension,
        ).transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside a Transformer block."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(d_ff, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.activation(self.input_projection(inputs)))


class TransformerBlock(nn.Module):
    """Pre-layer-normalized Transformer block with residual connections."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual sublayers."""
        hidden_states = inputs + self.attention(self.norm1(inputs))
        return hidden_states + self.feed_forward(self.norm2(hidden_states))


class MiniGPT(nn.Module):
    """A compact decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0 or n_layers <= 0 or d_ff <= 0:
            raise ValueError("model dimensions must be positive")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for integer token IDs."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("input sequence exceeds context_length")

        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        return self.lm_head(self.final_norm(hidden_states))
