# encoder.py

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Store embedding dimension and max sequence length for debugging or potential reuse
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Create positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))

        # Compute sinusoidal positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions use sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions use cos

        # Register buffer to ensure it's part of the module but not trainable
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Ensure the input sequence length does not exceed max_len
        if x.size(1) > self.max_len:
            raise ValueError(f"Input sequence length ({x.size(1)}) exceeds max_len ({self.max_len}).")
        
        # Add positional encodings
        return x + self.pe[:, :x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super(EncoderBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multihead Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)


class ENCODER(nn.Module):
    def __init__(self, n_mels=80, hidden_dim=512, max_len=5000, encoder_layers=4, num_heads=8):
        super(ENCODER, self).__init__()
        # Store parameters
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.encoder_layers = encoder_layers
        self.num_heads = num_heads

        # 2 x Conv1D + GELU for feature space expansion
        self.conv1 = nn.Conv1d(self.n_mels, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
        self.activation = nn.GELU()

        # Positional Encoding
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim=self.hidden_dim, max_len=self.max_len)

        # Stack of Encoder Blocks
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(hidden_dim=self.hidden_dim, num_heads=self.num_heads)
            for _ in range(self.encoder_layers)
        ])

    def forward(self, mels_inputs):
        """
        Forward pass through the encoder.
        Args:
            mels_inputs (torch.Tensor): Input spectrograms of shape (batch_size, n_mels, time_steps).
        
        Returns:
            torch.Tensor: Output from the encoder layers.
        """
        # Input: (batch_size, n_mels, time_steps)
        # Step 1: Expand feature space
        x = self.conv1(mels_inputs)  # (batch_size, hidden_dim, time_steps)
        x = self.activation(x)
        x = self.conv2(x)  # (batch_size, hidden_dim, time_steps)

        # Transpose to match the shape expected by the Transformer (batch_size, time_steps, hidden_dim)
        x = x.transpose(1, 2)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x
