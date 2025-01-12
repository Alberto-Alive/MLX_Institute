# encoder_split.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_positional_encoding(max_seq_len, embedding_dim):
    """
    Generates sinusoidal positional encodings.
    """
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(max_seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class Magic(nn.Module):
    """
    Custom Transformer Encoder Layer with Multi-Head Attention and Feed-Forward Network.
    """
    def __init__(self, embedding_dim, num_heads):
        super(Magic, self).__init__()

        # Parameters
        self.embedding_dim = embedding_dim          # Model dimensionality
        self.num_heads = num_heads                  # Number of attention heads
        self.head_dim = embedding_dim // num_heads  # Dimensionality per head

        # Linear projections for Multi-Head Attention
        self.W_Q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)

        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),  # Expand dimensionality
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim)   # Project back
        )

        # Dropout Layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        Forward pass for the Magic layer.
        Args:
            inputs: Tensor of shape [batch_size, seq_length, embedding_dim]
        Returns:
            Tensor of shape [batch_size, seq_length, embedding_dim]
        """
        # Multi-Head Self-Attention Sublayer
        # Pre-Norm Layer Normalization
        inputs_norm = self.layer_norm1(inputs)

        # Linear Projections
        Q = self.W_Q(inputs_norm)  # [batch_size, seq_length, embedding_dim]
        K = self.W_K(inputs_norm)  # [batch_size, seq_length, embedding_dim]
        V = self.W_V(inputs_norm)  # [batch_size, seq_length, embedding_dim]

        # Reshape for Multi-Head Attention
        batch_size, seq_length, _ = Q.size()
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_length, seq_length]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attn_output = torch.matmul(attn_probs, V)  # [batch_size, num_heads, seq_length, head_dim]

        # Concatenate Heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)  # [batch_size, seq_length, embedding_dim]

        # Final Linear Layer
        attn_output = self.W_O(attn_output)  # [batch_size, seq_length, embedding_dim]

        # Dropout and Residual Connection
        attn_output = self.dropout(attn_output)
        attn_output = inputs + attn_output  # Residual Connection

        # Position-wise Feed-Forward Network Sublayer
        # Pre-Norm Layer Normalization
        attn_output_norm = self.layer_norm2(attn_output)

        # Feed-Forward Network
        ffn_output = self.ffn(attn_output_norm)  # [batch_size, seq_length, embedding_dim]
        ffn_output = self.dropout(ffn_output)

        # Residual Connection
        out = attn_output + ffn_output  # [batch_size, seq_length, embedding_dim]

        return out


class ENCODER(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=8, max_sequence_length=784):
        """
        Transformer Encoder that includes image patch embedding and multiple Magic layers.
        Args:
            embedding_dim (int): Dimension of embeddings.
            num_heads (int): Number of attention heads.
            max_sequence_length (int): Maximum number of patches (sequence length).
        """
        super(ENCODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Image Patch Embedding: Converts flattened patches to embedding vectors
        self.image_embedding = nn.Linear(3 * 8 * 8, self.embedding_dim)  # 3 channels, 8x8 pixels

        # Positional Encoding
        self.pos_encoding = get_positional_encoding(self.max_sequence_length, self.embedding_dim)
        self.register_buffer("pos_encoding_buffer", self.pos_encoding)  # Register as buffer to avoid updating

        # Transformer Encoder Layers (Magic)
        self.magics = nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])

    def forward(self, inputs):
        """
        Forward pass for the ENCODER.
        Args:
            inputs: Tensor of shape [batch_size, 784, 3, 8, 8]
        Returns:
            Tensor of shape [batch_size, 784, embedding_dim]
        """
        batch_size, num_patches, channels, height, width = inputs.size()

        # Flatten patches
        patches = inputs.view(batch_size, num_patches, -1)  # [batch_size, 784, 3*8*8=192]

        # Embed patches
        embeddings = self.image_embedding(patches)  # [batch_size, 784, embedding_dim]

        # Add Positional Encoding
        pos_enc = self.pos_encoding_buffer[:num_patches, :].unsqueeze(0).repeat(batch_size, 1, 1).to(inputs.device)  # [batch_size, 784, embedding_dim]
        embeddings = embeddings + pos_enc  # [batch_size, 784, embedding_dim]

        # Pass through Magic layers
        for magic in self.magics:
            embeddings = magic(embeddings)  # [batch_size, 784, embedding_dim]

        return embeddings
