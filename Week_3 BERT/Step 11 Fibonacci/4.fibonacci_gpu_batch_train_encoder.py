from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import math
import json
import random
from tqdm import tqdm
import time
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the models directory exists
os.makedirs("./models/", exist_ok=True)

# Load the training data and vocabulary
with open("./data/training_data.json", "r") as f:
    loaded_training_data = json.load(f)

with open("./data/vocabulary.json", "r") as f:
    loaded_vocab = json.load(f)

# Create vocab-to-index and index-to-vocab mappings
vocab = {word: idx for word, idx in loaded_vocab.items()}
vocab_size = len(vocab)
idx_to_vocab = {idx: word for word, idx in vocab.items()}

torch.manual_seed(29)

# Function to generate Fibonacci numbers and create positional encoding
def generate_fibonacci(max_length, embedding_dim):
    fib = [0, 1]
    for _ in range(2, max_length + 1):
        fib.append(fib[-1] + fib[-2])
    max_fib = fib[-1] if fib[-1] != 0 else 1
    fib_normalized = [f / max_fib for f in fib[:max_length]]
    fib_matrix = []
    for pos in fib_normalized:
        fib_features = []
        for i in range(embedding_dim):
            fib_features.append(pos * math.sin(math.pi * i / embedding_dim))
        fib_matrix.append(fib_features)
    return torch.tensor(fib_matrix, dtype=torch.float32)

# Fibonacci Positional Encoding Module
class FibonacciPositionalEncoding(torch.nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(FibonacciPositionalEncoding, self).__init__()
        self.pos_encoding = generate_fibonacci(max_length, embedding_dim).unsqueeze(0)  # Shape: [1, max_length, embedding_dim]
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_length, embedding_dim]
        Returns:
            Tensor with positional encoding added: [batch_size, seq_length, embedding_dim]
        """
        seq_length = x.size(1)
        return x + self.pos_encoding[:, :seq_length, :].to(x.device)

# Updated Magic class with ReLU after projections
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
        Q = F.relu(self.W_Q(inputs))  # Add ReLU after projection
        K = F.relu(self.W_K(inputs))
        V = F.relu(self.W_V(inputs))

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

# Updated BERT class with Fibonacci Positional Encoding
class BERT(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=8, max_sequence_length=128):
        super(BERT, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)  # Updated for loaded vocab size
        
        # Fibonacci Positional Encoding
        self.fib_pos_encoding = FibonacciPositionalEncoding(max_sequence_length, embedding_dim)
        
        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(12)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)  # Linear layer to predict vocab size
    
    def forward(self, inputs):
        embs = self.emb(inputs)  # Embeddings for tokens
        embs = self.fib_pos_encoding(embs)  # Add Fibonacci positional encoding

        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)  # Shape: [batch_size, seq_length, vocab_size]
        probs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_length, vocab_size]
        return probs

# Initialize model, optimizer, and hyperparameters
B = BERT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.Adam(B.parameters(), lr=0.001)

# CustomDataset remains unchanged
class CustomDataset(Dataset):
    def __init__(self, training_data):
        self.data = training_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, targets = self.data[idx]
        inputs = torch.tensor(sequence, dtype=torch.long)  # Input sequence as tensor
        target_positions = torch.tensor([target[0] for target in targets], dtype=torch.long)  # Positions of target words
        target_words = torch.tensor([target[1] for target in targets], dtype=torch.long)  # Target words
        return inputs, target_positions, target_words

# Create the DataLoader
batch_size = 250  # Set your desired batch size
train_dataset = CustomDataset(loaded_training_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(50):
    total_loss = 0
    correct_predictions = 0
    total_masked_tokens = 0
    start_time = time.time()
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}") as pbar:
        for inputs, target_positions, target_words in train_loader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            target_positions = target_positions.to(device)
            target_words = target_words.to(device)

            # Get model outputs
            outputs = B(inputs)  # Shape: [batch_size, seq_length, vocab_size]

            # Gather only target positions for computing the loss
            batch_size_current = outputs.size(0)
            num_masked_tokens = target_positions.size(1)
            masked_outputs = outputs[torch.arange(batch_size_current).unsqueeze(1), target_positions]  # Shape: [batch_size, num_masked_tokens, vocab_size]
            masked_outputs = masked_outputs.view(-1, vocab_size)  # Shape: [batch_size * num_masked_tokens, vocab_size]
            masked_targets = target_words.view(-1)  # Shape: [batch_size * num_masked_tokens]

            # Compute loss only on masked tokens
            loss = F.nll_loss(masked_outputs, masked_targets, ignore_index=-100)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(B.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                preds = masked_outputs.argmax(dim=-1)
                mask = masked_targets != -100
                correct_predictions += (preds[mask] == masked_targets[mask]).sum().item()
                total_masked_tokens += mask.sum().item()

            pbar.update(1)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4%}, Execution time = {exec_time:.4f} seconds")

    if accuracy > 0.95:
        print("Stopping early as accuracy surpassed 95%.")
        break

# Save the model state
model_save_path = f"./models/bert_model_epoch_{epoch + 1}_loss_{avg_loss:.4f}_acc_{accuracy:.4%}_time_{exec_time:.4f}_batch_{batch_size}.pt"
torch.save(B.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
