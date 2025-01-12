# decoder_split.py

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

class CrossAttention(nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = decoder_embedding_dim  # Output embedding dimension
        self.head_dim = self.embedding_dim // num_heads

        assert self.embedding_dim % num_heads == 0, "decoder_embedding_dim must be divisible by num_heads"

        # Linear layers for queries, keys, and values
        self.W_Q = nn.Linear(decoder_embedding_dim, decoder_embedding_dim)
        self.W_K = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.W_V = nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.W_O = nn.Linear(decoder_embedding_dim, decoder_embedding_dim)

        self.layer_norm = nn.LayerNorm(decoder_embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_embedding_dim, decoder_embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(decoder_embedding_dim * 4, decoder_embedding_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, decoder_embs, encoder_embs):
        """
        Args:
            decoder_embs: [batch_size, tgt_len, decoder_embedding_dim]
            encoder_embs: [batch_size, src_len, encoder_embedding_dim]
        Returns:
            out: [batch_size, tgt_len, decoder_embedding_dim]
        """
        # Apply layer normalization to decoder embeddings
        decoder_embs_norm = self.layer_norm(decoder_embs)

        batch_size, tgt_len, _ = decoder_embs_norm.size()
        _, src_len, _ = encoder_embs.size()

        # Compute queries, keys, and values
        Q = self.W_Q(decoder_embs_norm)  # [batch_size, tgt_len, decoder_embedding_dim]
        K = self.W_K(encoder_embs)        # [batch_size, src_len, decoder_embedding_dim]
        V = self.W_V(encoder_embs)        # [batch_size, src_len, decoder_embedding_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, tgt_len, head_dim]
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, src_len, head_dim]
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, src_len, head_dim]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, tgt_len, src_len]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, tgt_len, src_len]
        head_outputs = torch.matmul(attn_probs, V)  # [batch_size, num_heads, tgt_len, head_dim]

        # Concatenate heads and pass through the output linear layer
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embedding_dim)  # [batch_size, tgt_len, embedding_dim]
        attn_output = self.W_O(concat_output)  # [batch_size, tgt_len, embedding_dim]
        attn_output = self.dropout(attn_output)

        # Residual connection
        out = decoder_embs + attn_output  # [batch_size, tgt_len, embedding_dim]

        # Apply feed-forward network with another residual connection
        out_norm = self.layer_norm(out)  # Pre-Norm before FFN
        ffn_output = self.ffn(out_norm)  # [batch_size, tgt_len, embedding_dim]
        ffn_output = self.dropout(ffn_output)
        out = out + ffn_output  # [batch_size, tgt_len, embedding_dim]

        return out

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
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, seq_length, embedding_dim]
        Returns:
            out: [batch_size, seq_length, embedding_dim]
        """
        inputs_norm = self.layer_norm(inputs)
        batch_size, seq_length, _ = inputs_norm.size()
        device = inputs.device

        Q = self.W_Q(inputs_norm)  # [batch_size, seq_length, embedding_dim]
        K = self.W_K(inputs_norm)  # [batch_size, seq_length, embedding_dim]
        V = self.W_V(inputs_norm)  # [batch_size, seq_length, embedding_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_length, seq_length]

        # Causal Masking
        causal_mask = torch.triu(torch.ones((1, 1, seq_length, seq_length), device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attn_probs = F.dropout(attn_probs, p=0.1, training=self.training)
        head_outputs = torch.matmul(attn_probs, V)  # [batch_size, num_heads, seq_length, head_dim]

        # Concatenate heads and pass through the output linear layer
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)  # [batch_size, seq_length, embedding_dim]
        attn_output = self.W_O(concat_output)  # [batch_size, seq_length, embedding_dim]
        attn_output = self.dropout(attn_output)

        # Residual connection
        out = inputs + attn_output  # [batch_size, seq_length, embedding_dim]
        return out

class DecoderLayer(nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(decoder_embedding_dim, num_heads)
        self.cross_attention = CrossAttention(encoder_embedding_dim, decoder_embedding_dim, num_heads)

    def forward(self, decoder_embs, encoder_embs):
        """
        Args:
            decoder_embs: [batch_size, tgt_len, decoder_embedding_dim]
            encoder_embs: [batch_size, src_len, encoder_embedding_dim]
        Returns:
            decoder_embs: [batch_size, tgt_len, decoder_embedding_dim]
        """
        decoder_embs = self.self_attention(decoder_embs)
        decoder_embs = self.cross_attention(decoder_embs, encoder_embs)
        return decoder_embs

class DECODER(nn.Module):
    def __init__(self, vocab_size=10, decoder_embedding_dim=64, encoder_embedding_dim=64, num_heads=4, max_sequence_length=6):
        super(DECODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.decoder_embedding_dim = decoder_embedding_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.embedding_dim = decoder_embedding_dim  # Define embedding_dim for consistency
        self.num_heads = num_heads
        self.emb = nn.Embedding(vocab_size, decoder_embedding_dim)
        
        # Positional encoding
        self.pos_encoding = get_positional_encoding(self.max_sequence_length, self.embedding_dim)
        self.register_buffer("pos_encoding_buffer", self.pos_encoding)  # Register as buffer to avoid updating

        # Transformer Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                encoder_embedding_dim=self.encoder_embedding_dim, 
                decoder_embedding_dim=self.decoder_embedding_dim, 
                num_heads=self.num_heads
            ) for _ in range(4)
        ])

        # Final linear layer to predict vocab size
        self.vocab = nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, target_sequence, encoder_output):
        """
        Args:
            target_sequence: [batch_size, max_seq_length]
            encoder_output: [batch_size, src_len, encoder_embedding_dim]
        Returns:
            logits: [batch_size, max_seq_length, vocab_size]
        """
        # Embed the target tokens
        embs = self.emb(target_sequence)  # [batch_size, max_seq_length, decoder_embedding_dim]

        batch_size, seq_length = target_sequence.size()
        device = embs.device

        # Add positional encoding
        pos_encodings = self.pos_encoding_buffer[:seq_length, :].unsqueeze(0).repeat(batch_size, 1, 1).to(device)  # [batch_size, seq_length, embedding_dim]
        decoder_embs = embs + pos_encodings  # [batch_size, seq_length, embedding_dim]

        # Pass through decoder layers
        for layer in self.layers:
            decoder_embs = layer(decoder_embs, encoder_output)  # [batch_size, seq_length, embedding_dim]

        # Map to vocabulary
        logits = self.vocab(decoder_embs)  # [batch_size, seq_length, vocab_size]
        return logits
