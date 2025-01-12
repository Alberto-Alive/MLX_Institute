from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import math
import time
from torch.utils.data import DataLoader, Dataset


def get_positional_encoding(max_seq_len, embedding_dim):
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(max_seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class CrossAttention(torch.nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = decoder_embedding_dim  # Output embedding dimension
        self.head_dim = self.embedding_dim // num_heads

        assert self.embedding_dim % num_heads == 0, "decoder_embedding_dim must be divisible by num_heads"

        # Linear layers for queries, keys, and values
        self.W_Q = torch.nn.Linear(decoder_embedding_dim, decoder_embedding_dim)
        self.W_K = torch.nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.W_V = torch.nn.Linear(encoder_embedding_dim, decoder_embedding_dim)
        self.W_O = torch.nn.Linear(decoder_embedding_dim, decoder_embedding_dim)

        self.layer_norm = torch.nn.LayerNorm(decoder_embedding_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(decoder_embedding_dim, decoder_embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_embedding_dim * 4, decoder_embedding_dim)
        )
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, decoder_embs, encoder_embs):
        # Apply layer normalization to decoder embeddings
        decoder_embs_norm = self.layer_norm(decoder_embs)

        batch_size, tgt_len, _ = decoder_embs_norm.size()
        _, src_len, _ = encoder_embs.size()

        # Compute queries, keys, and values
        Q = self.W_Q(decoder_embs_norm)  # [batch_size, tgt_len, decoder_embedding_dim]
        K = self.W_K(encoder_embs)       # [batch_size, src_len, decoder_embedding_dim]
        V = self.W_V(encoder_embs)       # [batch_size, src_len, decoder_embedding_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)

        # Concatenate heads and pass through the output linear layer
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embedding_dim)
        attn_output = self.W_O(concat_output)
        attn_output = self.dropout(attn_output)

        # Residual connection
        out = decoder_embs + attn_output

        # Apply feed-forward network with another residual connection
        out_norm = self.layer_norm(out)
        ffn_output = self.ffn(out_norm)
        ffn_output = self.dropout(ffn_output)
        out = out + ffn_output

        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        assert self.embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.W_Q = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):
        inputs_norm = self.layer_norm(inputs)
        batch_size, sequence_length, _ = inputs_norm.size()
        device = inputs.device

        Q = self.W_Q(inputs_norm)
        K = self.W_K(inputs_norm)
        V = self.W_V(inputs_norm)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Mask
        batch_size, num_heads, sequence_length, _ = attn_scores.size()
        causal_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)

        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)

        attn_output = self.W_O(concat_output)
        attn_output = self.dropout(attn_output)
        out = inputs + attn_output
        return out


class DecoderLayer(torch.nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(decoder_embedding_dim, num_heads)
        self.cross_attention = CrossAttention(encoder_embedding_dim, decoder_embedding_dim, num_heads)

    def forward(self, decoder_embs, encoder_embs):
        decoder_embs = self.self_attention(decoder_embs)
        decoder_embs = self.cross_attention(decoder_embs, encoder_embs)
        return decoder_embs


class ADECODER(torch.nn.Module):
    def __init__(self, vocab_size=10, decoder_embedding_dim=64, encoder_embedding_dim=64, num_heads=4, max_sequence_length=18):
        super(ADECODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.decoder_embedding_dim = decoder_embedding_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.embedding_dim = decoder_embedding_dim  # Define embedding_dim for consistency
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, decoder_embedding_dim)
        # Positional encoding
        self.register_buffer("pos_encoding", get_positional_encoding(self.max_sequence_length, self.embedding_dim))
        self.layers = torch.nn.ModuleList([DecoderLayer(decoder_embedding_dim=self.decoder_embedding_dim, encoder_embedding_dim=self.encoder_embedding_dim, num_heads=self.num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)  # Linear layer to predict vocab size

    def forward(self, target_sequence, encoder_output):
        embs = self.emb(target_sequence)  # Embeddings for tokens
        batch_size, seq_length = target_sequence.size()
        device = embs.device  # Get the device from embeddings

        # Ensure positional encodings are on the correct device
        pos_encodings = self.pos_encoding[:seq_length, :].to(device)
        decoder_embs = embs + pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            decoder_embs = layer(decoder_embs, encoder_output)

        logits = self.vocab(decoder_embs)
        return logits
