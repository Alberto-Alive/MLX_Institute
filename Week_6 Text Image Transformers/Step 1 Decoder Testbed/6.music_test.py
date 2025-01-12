import torch
import torch.nn.functional as F
import math
import string

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your vocabulary and genre mapping
loaded_vocab = {letter: idx + 1 for idx, letter in enumerate(string.ascii_uppercase)} 
idx_to_vocab = {idx: word for word, idx in loaded_vocab.items()}
genre_to_idx = {"rock": 0, "jazz": 1, "classical": 2}

# Redefine the BERT model with genre conditioning
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
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).to(device)
        attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)
        
        out = self.W_O(concat_output)
        out = self.layer_norm(inputs + out)
        return out

class GenreEncoding(torch.nn.Module):
    def __init__(self, num_genres, embedding_dim):
        super(GenreEncoding, self).__init__()
        # Initialize a genre encoding matrix of shape [num_genres, embedding_dim]
        self.genre_encoding_matrix = torch.nn.Parameter(torch.randn(num_genres, embedding_dim))

    def forward(self, genre_idx):
        # Select the genre-specific encoding from the matrix based on genre index
        genre_encoding = self.genre_encoding_matrix[genre_idx]  # Shape: [batch_size, embedding_dim]
        return genre_encoding

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, num_genres, embedding_dim=64, num_heads=8, max_sequence_length=128):
        super(BERT, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize the genre encoding as before
        self.genre_encoding = GenreEncoding(num_genres, embedding_dim)
        
        # Rename positional encoding to match the saved model
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim))
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                pos_encoding_matrix[position, dimension] = math.sin(angle_rate) if dimension % 2 == 0 else math.cos(angle_rate)
        self.register_buffer("positional_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)
    
    def forward(self, inputs, genre_idx):
        embs = self.emb(inputs)  # Embeddings for tokens
        pos_encodings = self.positional_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings

        # Add genre-specific encoding
        genre_encoding = self.genre_encoding(genre_idx).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        embs = embs + genre_encoding

        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)
        return F.log_softmax(logits, dim=-1)


# Function to test multi-step prediction
def test_model_multi_step(model, input_sequence, genre, steps=5):
    input_indices = [loaded_vocab.get(token, 0) for token in input_sequence]  # Use 0 for unknown tokens
    inputs = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    genre_idx = torch.tensor([genre_to_idx[genre]], dtype=torch.long).to(device)
    
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            output = model(inputs, genre_idx)
            next_token_logits = output[:, -1, :]
            next_token_index = next_token_logits.argmax(dim=-1).item()
            predicted_token = idx_to_vocab.get(next_token_index, "<UNK>")
            predictions.append(predicted_token)
            
            # Append predicted token and update input sequence
            input_indices.append(next_token_index)
            inputs = torch.tensor(input_indices[-len(input_sequence):], dtype=torch.long).unsqueeze(0).to(device)
    
    return predictions

# Testing class for note generation based on genre
class NoteGenerator:
    def __init__(self, model, genres):
        self.model = model
        self.genres = genres

    def set_genre(self, genre):
        if genre in self.genres:
            self.genre = genre
        else:
            raise ValueError("Genre not recognized.")

    def generate(self, input_sequence, steps=10):
        return test_model_multi_step(self.model, input_sequence, self.genre, steps)

# Load the model and initialize genre list
vocab_size = len(loaded_vocab) + 1
num_genres = len(genre_to_idx)
model = BERT(vocab_size=vocab_size, num_genres=num_genres).to(device)
model.load_state_dict(torch.load("./models/genre_conditioned_model.pt", map_location=device, weights_only=True))
model.eval()

# Instantiate NoteGenerator and test
generator = NoteGenerator(model, genres=list(genre_to_idx.keys()))
generator.set_genre("rock")
starting_sequence = ["T", "P", "C"]

# Generate notes
generated_notes = generator.generate(starting_sequence, steps=6)
print(f"Generated notes in rock genre: {generated_notes}")
