from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import math
import json
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

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
B = BERT(vocab_size=vocab_size)
optimizer = torch.optim.Adam(B.parameters(), lr=0.001)



class MaskedLanguageModelDataset(Dataset):
    def __init__(self, training_data, max_seq_length):
        self.training_data = training_data
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        masked_sequence, target_tokens = self.training_data[idx]
        seq_length = len(masked_sequence)

        # Create target tensor initialized with -100 (ignored index in loss)
        target_tensor = torch.full((seq_length,), -100, dtype=torch.long)

        # Set the target tokens at the masked positions
        for position, target_token_id in target_tokens:
            if position < seq_length:
                target_tensor[position] = target_token_id

        return torch.tensor(masked_sequence), target_tensor

def collate_fn(batch):
    masked_sequences, target_tensors = zip(*batch)
    padded_sequences = pad_sequence([seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq) for seq in masked_sequences], 
                                    batch_first=True, padding_value=4)
    padded_targets = pad_sequence(target_tensors, batch_first=True, padding_value=-100)
    return padded_sequences, padded_targets




max_seq_length = 128
batch_size = 100
dataset = MaskedLanguageModelDataset(loaded_training_data, max_seq_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


for epoch in range(500):
    total_loss = 0
    correct_predictions = 0
    total_masked_tokens = 0

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}") as pbar:
        for masked_sequences, target_tokens in data_loader:
            optimizer.zero_grad()
            outputs = B(masked_sequences)  # Outputs shape: [batch_size, seq_length, vocab_size]

            # Flatten outputs and targets for loss computation
            outputs = outputs.view(-1, vocab_size)        # Shape: [batch_size * seq_length, vocab_size]
            targets = target_tokens.view(-1)              # Shape: [batch_size * seq_length]

            # Compute loss, ignoring the positions with target -100
            loss = F.nll_loss(outputs, targets, ignore_index=-100)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            # Calculate accuracy
            with torch.no_grad():
                preds = outputs.argmax(dim=-1)
                mask = targets != -100
                correct_predictions += (preds[mask] == targets[mask]).sum().item()
                total_masked_tokens += mask.sum().item()

            pbar.update(1)

    # Compute average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0

    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4%}")

    if accuracy > 0.95:
        print("Stopping early as accuracy surpassed 95%.")
        break





















































# class MaskedLanguageModelDataset(Dataset):
#     def __init__(self, training_data):
#         self.training_data = training_data

#     def __len__(self):
#         return len(self.training_data)

#     def __getitem__(self, idx):
#         masked_sequence, target_tokens = self.training_data[idx]
#         return torch.tensor(masked_sequence), target_tokens

# # Initialize dataset and dataloader
# batch_size = 100
# dataset = MaskedLanguageModelDataset(loaded_training_data)

# def collate_fn(batch):
#     masked_sequences, target_tokens_list = zip(*batch)  # Separate inputs and targets
#     padded_sequences = pad_sequence([seq.clone() if isinstance(seq, torch.Tensor) else torch.tensor(seq) for seq in masked_sequences], 
#                                     batch_first=True, padding_value=4)
#     return padded_sequences, target_tokens_list

# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)





# for epoch in range(500):  # Set a more realistic epoch limit
#     total_loss = 0
#     correct_predictions = 0
#     total_masked_tokens = 0
    
#     with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}") as pbar:
#         for masked_sequence, target_tokens_list  in data_loader:
#             optimizer.zero_grad()

#             # Convert masked_sequence to tensor and add batch dimension
#             outputs = B(masked_sequence)  # Model's predicted logits for each token in the sequence

#             # Initialize loss for this sample
#             sample_loss = 0

#             for i, target_tokens in enumerate(target_tokens_list):
#                 for position, target_token_id in target_tokens:
#                     # Get the model's prediction at the masked position
#                     predicted_logits = outputs[i, position, :]

#                     # Convert the target token id to a tensor
#                     target_token_tensor = torch.tensor([target_token_id])

#                     # Calculate the loss only for this masked position
#                     loss = F.nll_loss(predicted_logits.unsqueeze(0), target_token_tensor)
#                     sample_loss += loss

#                     # Check if prediction is correct
#                     predicted_token_id = predicted_logits.argmax(dim=-1).item()
#                     if predicted_token_id == target_token_id:
#                         correct_predictions += 1

#             # Accumulate the total number of masked tokens for accuracy calculation
#             total_masked_tokens += len(target_tokens)

#             # Backpropagation
#             sample_loss.backward()
#             optimizer.step()

#             # Accumulate total loss
#             total_loss += sample_loss.item()
#             pbar.update(1)

#     # Print average loss and correct prediction percentage per epoch
#     avg_loss = total_loss / len(loaded_training_data)
#     accuracy = correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0

#     print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4%}")

#     # Early stopping condition (optional)
#     if accuracy > 0.95:  # Stop if accuracy surpasses a threshold
#         print("Stopping early as accuracy surpassed 95%.")
#         break


# # Save the trained model's state_dict
# model_save_path = "./models/bert_model.pt"
# torch.save(B.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")



