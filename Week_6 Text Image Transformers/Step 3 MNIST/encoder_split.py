from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import math


def get_positional_encoding(max_seq_len, embedding_dim):
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(max_seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


class Magic(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Magic, self).__init__()

        # Parameters
        self.embedding_dim = embedding_dim          # Model dimensionality
        self.num_heads = num_heads                  # Number of attention heads
        self.head_dim = embedding_dim // num_heads  # Dimensionality per head

        # Title: Linear Projections for Multi-Head Attention
        # Description: Layers to project inputs into queries (Q), keys (K), and values (V), and to combine the outputs
        self.W_Q = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        # Title: Layer Normalization Layers
        # Description: Apply layer normalization before attention and feed-forward sublayers (Pre-Norm)
        self.layer_norm1 = torch.nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(self.embedding_dim)

        # Title: Position-wise Feed-Forward Network
        # Description: A two-layer feed-forward network with ReLU activation in between
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim * 4),  # Expand dimensionality
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim)   # Project back to original dimensionality
        )

        # Title: Dropout Layer
        # Description: Regularization to prevent overfitting
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):
        # Title: Multi-Head Self-Attention Sublayer
        # Description: Computes self-attention over the input sequence

        # Step 1: Pre-Norm Layer Normalization
        inputs_norm = self.layer_norm1(inputs)

        # Step 2: Linear Projections to Obtain Q, K, V
        batch_size, sequence_length, _ = inputs_norm.size()
        Q = self.W_Q(inputs_norm)  # Queries
        K = self.W_K(inputs_norm)  # Keys
        V = self.W_V(inputs_norm)  # Values

        # Step 3: Reshape and Transpose for Multi-Head Attention
        # Reshape to (batch_size, num_heads, sequence_length, head_dim)
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 4: Scaled Dot-Product Attention
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Apply softmax to obtain attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        # Compute weighted sum of values
        head_outputs = torch.matmul(attn_probs, V)

        # Step 5: Concatenate Heads and Final Linear Projection
        # Transpose and reshape back to (batch_size, sequence_length, embedding_dim)
        concat_output = head_outputs.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.embedding_dim
        )
        # Final linear layer to combine heads
        attn_output = self.W_O(concat_output)

        # Step 6: Apply Dropout
        attn_output = self.dropout(attn_output)

        # Step 7: Residual Connection after Attention
        attn_output = inputs + attn_output  # Add the original input (residual connection)

        # Title: Position-wise Feed-Forward Network Sublayer
        # Description: Applies a feed-forward network to each position independently

        # Step 8: Pre-Norm Layer Normalization
        attn_output_norm = self.layer_norm2(attn_output)

        # Step 9: Feed-Forward Network
        ffn_output = self.ffn(attn_output_norm)
        # Apply dropout for regularization
        ffn_output = self.dropout(ffn_output)

        # Step 10: Residual Connection after Feed-Forward Network
        out = attn_output + ffn_output  # Add the input to the FFN output (residual connection)

        # Final output of the encoder layer
        return out


class ENCODER(torch.nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4, max_sequence_length=16):
        super(ENCODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.register_buffer("pos_encoding", get_positional_encoding(self.max_sequence_length, self.embedding_dim))

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])
    
    def forward(self, inputs):
        batch_size, group_size, embedding_dim = inputs.size()
        device = inputs.device  # Get the device from the input tensor

        # Ensure positional encodings are on the correct device
        pos_encodings = self.pos_encoding[:group_size, :].to(device)
        pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, group_size, embedding_dim]
        embs = inputs + pos_encodings

        for magic in self.magics:
            embs = magic(embs)
        return embs
    
  