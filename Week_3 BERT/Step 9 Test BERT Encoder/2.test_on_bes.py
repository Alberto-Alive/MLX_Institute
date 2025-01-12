import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

# Define the vocabulary with a padding token
vocab = {"A": 0, "B": 1, "C": 2, "D": 3, "[MASK]": 4, "[PAD]": 5}
idx_to_token = {v: k for k, v in vocab.items()}

# Example pattern set (train) with variable lengths
train = [
    (["A", "?", "B"], "A"),
    (["C", "B", "?"], "B"),
    (["B", "C", "?"], "C"),
    (["A", "?", "A"], "A"),
    (["B", "A", "?"], "A"),
    (["C", "?", "B"], "C"),
    (["A", "B", "?", "A"], "B"),
    (["C", "A", "C", "?"], "A"),
    (["B", "?", "B", "A"], "A"),
    (["A", "B", "C", "?"], "D"),
    (["?", "B", "C", "D"], "A"),
    (["A", "?", "C", "D"], "B"),
    (["A", "A", "?", "B", "B"], "A"),
    (["C", "C", "B", "?", "B"], "C"),
    (["B", "A", "B", "?", "A"], "B")
]

# Preprocess data with padding
def preprocess_data(train_data, vocab):
    inputs, targets = [], []
    for sequence, target in train_data:
        input_indices = [vocab.get(token, vocab["[MASK]"]) if token != "?" else vocab["[MASK]"] for token in sequence]
        target_index = vocab[target]
        inputs.append(torch.tensor(input_indices))  # Convert each sequence to a tensor
        targets.append(target_index)
    # Pad sequences to the length of the longest sequence
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["[PAD]"])
    targets = torch.tensor(targets)  # Targets don't need padding
    return inputs, targets

# Preprocess the training data
inputs, targets = preprocess_data(train, vocab)

# Define a custom transformer model with embedding and mask handling
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
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)

        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim))
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                pos_encoding_matrix[position, dimension] = math.sin(angle_rate) if dimension % 2 == 0 else math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embs = self.emb(inputs)
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings

        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Initialize the model
vocab_size = 6
orig_vocab_len =16000
model = BERT(vocab_size=orig_vocab_len)

# Load the saved model weights
model_filepath = "./models/better_name.pt"  # Replace with your actual file path
state_dict = torch.load(model_filepath, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Forward pass and evaluation
correct = 0
total = len(inputs)

with torch.no_grad():
    for i in range(total):
        input_sequence = inputs[i].unsqueeze(0)  # Add batch dimension
        target = targets[i]

        # Pass through the model
        output = model(input_sequence)
        
        # Find the index of the masked token and get the prediction
        mask_index = (input_sequence == vocab["[MASK]"]).nonzero(as_tuple=True)[1]
        if mask_index.numel() > 0:  # Ensure there is a mask token
            predicted_token = output[0, mask_index, :].argmax(dim=-1).item()
        else:
            predicted_token = -1  # If no mask found, set an invalid token

        # Compare with the target
        if predicted_token == target.item():
            correct += 1

        # Print results for each example
        input_tokens = [idx_to_token[idx.item()] for idx in input_sequence[0] if idx != vocab["[PAD]"]]
        print(f"Input: {''.join(input_tokens)}")
        print(f"Expected: {idx_to_token[target.item()]}, Predicted: {idx_to_token[predicted_token]}\n")

# Calculate accuracy
accuracy = correct / total
print(f"Accuracy: {accuracy * 100:.2f}%")
