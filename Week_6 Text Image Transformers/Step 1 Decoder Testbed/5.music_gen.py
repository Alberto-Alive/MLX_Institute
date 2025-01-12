import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate loaded_vocab with all letters in the alphabet
loaded_vocab = {letter: idx + 1 for idx, letter in enumerate(string.ascii_uppercase)}
idx_to_vocab = {idx: word for word, idx in loaded_vocab.items()}
genre_to_idx = {"rock": 0, "jazz": 1, "classical": 2}  # Example genre-to-index map

# Define dataset
class MusicDataset(Dataset):
    def __init__(self, training_data, vocab, genre_to_idx):
        self.data = training_data
        self.vocab = vocab
        self.genre_to_idx = genre_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target, genre = self.data[idx]
        inputs = torch.tensor([self.vocab[token] for token in sequence], dtype=torch.long)
        target = torch.tensor(self.vocab[target], dtype=torch.long)
        genre_idx = torch.tensor(self.genre_to_idx[genre], dtype=torch.long)
        return inputs, target, genre_idx

loaded_training_data = [
    # Rock genre patterns
    (["R", "O", "C", "K"], "R", "rock"),
    (["O", "C", "K", "R"], "O", "rock"),
    (["C", "K", "R", "O"], "C", "rock"),
    (["K", "R", "O", "C"], "K", "rock"),
    (["R", "K", "O", "C"], "R", "rock"),
    
    # Jazz genre patterns
    (["J", "A", "Z", "Z"], "J", "jazz"),
    (["A", "Z", "Z", "J"], "A", "jazz"),
    (["I", "J", "K", "L"], "I", "jazz"),
    (["J", "K", "L", "M"], "J", "jazz"),
    (["K", "L", "M", "N"], "K", "jazz"),
    
    # Classical genre patterns
    (["O", "P", "Q", "R"], "O", "classical"),
    (["P", "Q", "R", "S"], "P", "classical"),
    (["Q", "R", "S", "T"], "Q", "classical"),
    (["R", "S", "T", "U"], "R", "classical"),
    (["S", "T", "U", "V"], "S", "classical"),
]


# Define a Genre Encoding Matrix
class GenreEncoding(torch.nn.Module):
    def __init__(self, num_genres, embedding_dim):
        super(GenreEncoding, self).__init__()
        # Initialize a genre encoding matrix of shape [num_genres, embedding_dim]
        self.genre_encoding_matrix = torch.nn.Parameter(torch.randn(num_genres, embedding_dim))

    def forward(self, genre_idx):
        # Select the genre-specific encoding from the matrix based on genre index
        genre_encoding = self.genre_encoding_matrix[genre_idx]  # Shape: [batch_size, embedding_dim]
        return genre_encoding

# Define the BERT model with genre encoding
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
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)

        out = self.W_O(concat_output)
        out = self.layer_norm(inputs + out)
        return out

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, num_genres, embedding_dim=64, num_heads=8, max_sequence_length=128):
        super(BERT, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)
        self.genre_encoding = GenreEncoding(num_genres, embedding_dim)
        self.positional_encoding = torch.nn.Parameter(torch.randn(max_sequence_length, embedding_dim))  # Learnable positional encoding

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)
    
    def forward(self, inputs, genre_idx):
        embs = self.emb(inputs)  # Token embeddings for the input sequence
        pos_encodings = self.positional_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings  # Add positional encoding

        # Add genre-specific encoding
        genre_encoding = self.genre_encoding(genre_idx).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        embs = embs + genre_encoding  # Add genre encoding to each token position

        # Pass through transformer layers
        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)
        return F.log_softmax(logits, dim=-1)

# Initialize dataset, model, and training components
train_dataset = MusicDataset(loaded_training_data, loaded_vocab, genre_to_idx)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

vocab_size = len(loaded_vocab) + 1
num_genres = len(genre_to_idx)
model = BERT(vocab_size=vocab_size, num_genres=num_genres).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for inputs, target, genre_idx in train_loader:
        inputs, target, genre_idx = inputs.to(device), target.to(device), genre_idx.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, genre_idx)
        next_token_logits = outputs[:, -1, :]
        loss = criterion(next_token_logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        with torch.no_grad():
            predicted_index = next_token_logits.argmax(dim=-1)
            correct_predictions += (predicted_index == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader)
    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4%}")

# Save the model
torch.save(model.state_dict(), "./models/genre_conditioned_model.pt")
print("Model saved successfully.")
