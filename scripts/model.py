#!/usr/bin/env python3
"""
Custom Dad Joke Transformer - ~25M parameter decoder-only transformer
Trained from scratch on dad jokes using GPT-2's tokenizer.

Architecture:
- 6 transformer blocks
- 512 embedding dimension
- 8 attention heads
- 128 max sequence length
- Pre-norm (LayerNorm before attention/FFN)
- Weight-tied embedding and output head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DadJokeConfig:
    """Model configuration"""
    vocab_size = 50257       # GPT-2 tokenizer vocab size
    n_layers = 6             # Number of transformer blocks
    n_heads = 8              # Number of attention heads
    d_model = 512            # Embedding dimension
    d_ff = 2048              # Feed-forward hidden dimension (4x d_model)
    max_seq_len = 128        # Maximum sequence length
    dropout = 0.1            # Dropout rate

    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith('_') and k != 'to_dict'}


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask"""

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> Residual -> LN -> FFN -> Residual"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DadJokeTransformer(nn.Module):
    """
    Decoder-only transformer for dad joke generation.

    Uses GPT-2's tokenizer (50,257 vocab) but a custom small architecture.
    Weight tying between token embedding and output head reduces parameters.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DadJokeConfig()
        self.config = config

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final layer norm
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small normal distribution"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if targets provided, None otherwise
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8,
                 top_k=50, top_p=0.9, eos_token_id=None):
        """
        Autoregressive generation with top-k and top-p (nucleus) sampling.

        Args:
            input_ids: (1, seq_len) starting token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: keep only top-k logits
            top_p: nucleus sampling threshold
            eos_token_id: stop generation at this token

        Returns:
            (1, seq_len + generated) full sequence of token IDs
        """
        self.train(False)

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Account for weight tying (lm_head shares weights with token_emb)
    tied = model.token_emb.weight.numel()
    unique = total - tied
    return unique, trainable - tied


if __name__ == "__main__":
    config = DadJokeConfig()
    model = DadJokeTransformer(config)

    unique_params, trainable_params = count_parameters(model)
    print(f"Model: DadJokeTransformer")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Unique parameters: {unique_params:,}")
    print(f"  (with tied weights counted once)")

    # Test forward pass
    dummy = torch.randint(0, config.vocab_size, (2, 32))
    logits, _ = model(dummy)
    print(f"\n  Test forward: input {dummy.shape} -> logits {logits.shape}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"  Test generate: prompt {prompt.shape} -> output {generated.shape}")
    print("\nModel OK!")
