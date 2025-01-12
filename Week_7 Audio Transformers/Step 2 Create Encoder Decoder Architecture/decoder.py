# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        assert self.embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, causal_mask=None):
        """
        Args:
            x: [batch_size, seq_length, embedding_dim]
            causal_mask: Optional causal mask for autoregressive decoding
        Returns:
            out: [batch_size, seq_length, embedding_dim]
        """
        batch_size, seq_length, _ = x.size()
        Q = self.W_Q(x)  # [batch_size, seq_length, embedding_dim]
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, V)

        # Concatenate heads and pass through output linear layer
        concat_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        return self.W_O(concat_output)  # [batch_size, seq_length, embedding_dim]


class CrossAttention(nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.self_attention = SelfAttention(decoder_embedding_dim, num_heads)
        self.layer_norm = nn.LayerNorm(decoder_embedding_dim)

    def forward(self, decoder_embs, encoder_embs):
        # Pre-norm before cross-attention
        decoder_embs_norm = self.layer_norm(decoder_embs)
        return decoder_embs + self.self_attention(decoder_embs_norm, causal_mask=None)


class DecoderLayer(nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(decoder_embedding_dim, num_heads)
        self.cross_attention = CrossAttention(encoder_embedding_dim, decoder_embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_embedding_dim * 4, decoder_embedding_dim),
        )
        self.norm1 = nn.LayerNorm(decoder_embedding_dim)
        self.norm2 = nn.LayerNorm(decoder_embedding_dim)
        self.norm3 = nn.LayerNorm(decoder_embedding_dim)

    def forward(self, decoder_embs, encoder_embs, causal_mask=None):
        # Self-attention with pre-norm
        decoder_embs = decoder_embs + self.self_attention(self.norm1(decoder_embs), causal_mask=causal_mask)

        # Cross-attention with pre-norm
        decoder_embs = decoder_embs + self.cross_attention(self.norm2(decoder_embs), encoder_embs)

        # Feed-forward network with pre-norm
        decoder_embs = decoder_embs + self.feed_forward(self.norm3(decoder_embs))
        return decoder_embs


class DECODER(nn.Module):
    def __init__(self, vocab_size, decoder_embedding_dim, encoder_embedding_dim, num_heads, max_sequence_length, num_layers):
        super(DECODER, self).__init__()
        self.embedding_dim = decoder_embedding_dim
        self.emb = nn.Embedding(vocab_size, decoder_embedding_dim)

        # Learnable positional encodings
        self.positional_encoding = nn.Embedding(max_sequence_length, decoder_embedding_dim)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                encoder_embedding_dim=encoder_embedding_dim,
                decoder_embedding_dim=decoder_embedding_dim,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])

        # Final layer normalization (important in Whisper)
        self.final_norm = nn.LayerNorm(decoder_embedding_dim)

        # Vocabulary projection
        self.vocab_projection = nn.Linear(decoder_embedding_dim, vocab_size)

    def forward(self, target_sequence, encoder_output):
        """
        Args:
            target_sequence: [batch_size, max_seq_length]
            encoder_output: [batch_size, src_len, encoder_embedding_dim]
        Returns:
            logits: [batch_size, max_seq_length, vocab_size]
        """
        # Token embedding and positional encoding
        embs = self.emb(target_sequence)  # [batch_size, seq_length, decoder_embedding_dim]
        positions = torch.arange(0, target_sequence.size(1), device=target_sequence.device).unsqueeze(0)
        pos_encodings = self.positional_encoding(positions)
        decoder_embs = embs + pos_encodings

        # Causal mask for autoregressive decoding
        seq_len = target_sequence.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=target_sequence.device), diagonal=1).bool()

        # Pass through decoder layers
        for layer in self.layers:
            decoder_embs = layer(decoder_embs, encoder_output, causal_mask=causal_mask)

        # Apply final normalization and project to vocabulary
        decoder_embs = self.final_norm(decoder_embs)
        logits = self.vocab_projection(decoder_embs)
        return logits
