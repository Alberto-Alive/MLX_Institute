import torch
import torch.nn.functional as F
import json
import math

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your vocabulary and test data
loaded_vocab = {"A": 1, "B": 2, "C": 3}
idx_to_vocab = {idx: word for word, idx in loaded_vocab.items()}

# Re-define the BERT model class and submodules as before
class Magic(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Magic, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        
        self.W_Q = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
    
    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.size()
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # lego
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(device)
        attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))
        #lego


        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)

        out = self.W_O(concat_output)
        out = self.layer_norm(inputs + out)
        return out

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=8, max_sequence_length=128):
        super(BERT, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)  # Updated for loaded vocab size
        
        # Positional encoding
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim))
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                pos_encoding_matrix[position, dimension] = math.sin(angle_rate) if dimension % 2 == 0 else math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)  # Linear layer to predict vocab size
    
    def forward(self, inputs):
        embs = self.emb(inputs)  # Embeddings for tokens
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings

        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs


# Load the saved model
model_path = "./models/decoder.pt"  # replace with the actual model file path
vocab_size = len(loaded_vocab) + 1  # Adjust for index starting from 1
model = BERT(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Testing function to predict the next token
def test_model(model, input_sequence):
    # Convert input sequence to tensor of indices
    input_indices = [loaded_vocab[token] for token in input_sequence]
    inputs = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        output = model(inputs)  # Shape: [1, seq_length, vocab_size]
        next_token_logits = output[:, -1, :]  # Get logits for the next token prediction
        next_token_index = next_token_logits.argmax(dim=-1).item()  # Get index of predicted token
    
    # Convert index back to token
    predicted_token = idx_to_vocab.get(next_token_index, "<UNK>")
    return predicted_token

# Test the model with different input sequences
test_sequences = [
    ["A", "A", "B", "B", "C", "C"],
    ["A", "B", "B", "C", "C", "A"],
    ["B", "B", "C", "C", "A", "A"],
]

for seq in test_sequences:
    predicted_token = test_model(model, seq)
    print(f"Input Sequence: {seq} -> Predicted Next Token: {predicted_token}")



def test_model_multi_step(model, input_sequence, steps=5):
    input_indices = [loaded_vocab[token] for token in input_sequence]
    inputs = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(steps):
            output = model(inputs)
            next_token_logits = output[:, -1, :]
            next_token_index = next_token_logits.argmax(dim=-1).item()
            predicted_token = idx_to_vocab.get(next_token_index, "<UNK>")
            predictions.append(predicted_token)
            
            # Append the predicted token to the inputs for the next prediction
            input_indices.append(next_token_index)
            inputs = torch.tensor(input_indices[-len(input_sequence):], dtype=torch.long).unsqueeze(0).to(device)
    
    return predictions

# Test multi-step predictions
for seq in test_sequences:
    predicted_tokens = test_model_multi_step(model, seq, steps=6)
    print(f"Input Sequence: {seq} -> Predicted Next Tokens: {predicted_tokens}")
