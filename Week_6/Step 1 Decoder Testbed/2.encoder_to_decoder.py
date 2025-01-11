from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import math
import json
import random
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loaded_training_data = [
    (["A", "A", "B", "B", "C", "C"], "A"),  # After "A, A, B, B, C, C", the next token is "A"
    (["A", "B", "B", "C", "C", "A"], "A"),  # After "A, B, B, C, C, A", the next token is "A"
    (["B", "B", "C", "C", "A", "A"], "B"),  # After "B, B, C, C, A, A", the next token is "B"
    (["B", "C", "C", "A", "A", "B"], "B"),  # After "B, C, C, A, A, B", the next token is "B"
    (["C", "C", "A", "A", "B", "B"], "C"),  # After "C, C, A, A, B, B", the next token is "C"
    (["C", "A", "A", "B", "B", "C"], "C"),  # After "C, A, A, B, B, C", the next token is "C"
]
loaded_vocab = {"_NEXT": 0, "A": 1, "B": 2, "C": 3}





# Create vocab-to-index and index-to-vocab mappings
vocab = {word: idx for word, idx in loaded_vocab.items()}
vocab_size = len(vocab)
idx_to_vocab = {idx: word for word, idx in vocab.items()}

torch.manual_seed(29)

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

# Initialize model, optimizer, and hyperparameters
B = BERT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(B.parameters(), lr=0.001)

class CustomDataset(Dataset):
    def __init__(self, training_data, vocab):
        self.data = training_data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        # Convert sequence and target to indices
        inputs = torch.tensor([self.vocab[token] for token in sequence], dtype=torch.long)
        target = torch.tensor(self.vocab[target], dtype=torch.long)  # Single target token
        return inputs, target


train_dataset = CustomDataset(loaded_training_data, loaded_vocab)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Use batch_size=1 since we're processing one sequence at a time
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(500):
    total_loss = 0
    correct_predictions = 0
    start_time = time.time()
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}") as pbar:
        for inputs, target in train_loader:
            optimizer.zero_grad()

            # Move data to device
            inputs = inputs.to(device)
            target = target.to(device)

            # Get model outputs
            outputs = B(inputs)  # Shape: [1, seq_length, vocab_size]

            # Use the last output to predict the next token
            last_output = outputs[:, -1, :]  # Shape: [1, vocab_size]

            # Compute the loss
            loss = criterion(last_output, target)

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(B.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            with torch.no_grad():
                pred = last_output.argmax(dim=-1)
                correct_predictions += (pred == target).sum().item()

            pbar.update(1)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4%}, Execution time = {exec_time:.4f} seconds")

    if accuracy > 0.95:
        print("Stopping early as accuracy surpassed 95%.")
        break

# Save the model
model_save_path = f"./bert_model_epoch_{epoch + 1}_loss_{avg_loss:.4f}_acc_{accuracy:.4%}_time_{exec_time:.4f}.pt"
torch.save(B.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
