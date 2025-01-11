import torch
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Define GRL
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=0.5):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# Positional Encoding
def get_positional_encoding(max_seq_len, embedding_dim):
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    pos_encoding = torch.zeros(max_seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

# Define Self-Attention
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

        # Causal Mask
        causal_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)

        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)

        attn_output = self.W_O(concat_output)
        attn_output = self.dropout(attn_output)
        out = inputs + attn_output
        return out

# Define Cross-Attention
class CrossAttention(torch.nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = decoder_embedding_dim
        self.head_dim = self.embedding_dim // num_heads

        assert self.embedding_dim % num_heads == 0, "decoder_embedding_dim must be divisible by num_heads"

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
        decoder_embs_norm = self.layer_norm(decoder_embs)

        batch_size, tgt_len, _ = decoder_embs_norm.size()
        _, src_len, _ = encoder_embs.size()

        Q = self.W_Q(decoder_embs_norm)
        K = self.W_K(encoder_embs)
        V = self.W_V(encoder_embs)

        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)

        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embedding_dim)
        attn_output = self.W_O(concat_output)
        attn_output = self.dropout(attn_output)

        out = decoder_embs + attn_output

        # Feed-Forward Network
        out_norm = self.layer_norm(out)
        ffn_output = self.ffn(out_norm)
        ffn_output = self.dropout(ffn_output)
        out = out + ffn_output

        return out

# Define Decoder Layer
class DecoderLayer(torch.nn.Module):
    def __init__(self, encoder_embedding_dim, decoder_embedding_dim, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(decoder_embedding_dim, num_heads)
        self.cross_attention = CrossAttention(encoder_embedding_dim, decoder_embedding_dim, num_heads)

    def forward(self, decoder_embs, encoder_embs):
        decoder_embs = self.self_attention(decoder_embs)
        decoder_embs = self.cross_attention(decoder_embs, encoder_embs)
        return decoder_embs

# Define Decoder (Associative and Dissociative)
class ADECODER(torch.nn.Module):
    def __init__(self, vocab_size=12, decoder_embedding_dim=64, encoder_embedding_dim=64, num_heads=4, max_sequence_length=18):
        super(ADECODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.decoder_embedding_dim = decoder_embedding_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.embedding_dim = decoder_embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, decoder_embedding_dim)
        self.register_buffer("pos_encoding", get_positional_encoding(self.max_sequence_length, self.embedding_dim))
        self.layers = torch.nn.ModuleList([
            DecoderLayer(decoder_embedding_dim=self.decoder_embedding_dim, 
                        encoder_embedding_dim=self.encoder_embedding_dim, 
                        num_heads=self.num_heads) 
            for _ in range(4)
        ])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, target_sequence, encoder_output):
        embs = self.emb(target_sequence)  # [batch_size, seq_len, decoder_embedding_dim]
        batch_size, seq_length = target_sequence.size()
        device = embs.device

        pos_encodings = self.pos_encoding[:seq_length, :].to(device)
        decoder_embs = embs + pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            decoder_embs = layer(decoder_embs, encoder_output)

        logits = self.vocab(decoder_embs)
        return logits

class DDECODER(torch.nn.Module):
    def __init__(self, vocab_size=12, decoder_embedding_dim=64, encoder_embedding_dim=64, num_heads=4, max_sequence_length=18):
        super(DDECODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.decoder_embedding_dim = decoder_embedding_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.embedding_dim = decoder_embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, decoder_embedding_dim)
        self.register_buffer("pos_encoding", get_positional_encoding(self.max_sequence_length, self.embedding_dim))
        self.layers = torch.nn.ModuleList([
            DecoderLayer(decoder_embedding_dim=self.decoder_embedding_dim, 
                        encoder_embedding_dim=self.encoder_embedding_dim, 
                        num_heads=self.num_heads) 
            for _ in range(4)
        ])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, target_sequence, encoder_output):
        embs = self.emb(target_sequence)  # [batch_size, seq_len, decoder_embedding_dim]
        batch_size, seq_length = target_sequence.size()
        device = embs.device

        pos_encodings = self.pos_encoding[:seq_length, :].to(device)
        decoder_embs = embs + pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            decoder_embs = layer(decoder_embs, encoder_output)

        logits = self.vocab(decoder_embs)
        return logits

# Define Encoder
class Magic(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Magic, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.W_Q = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self.layer_norm1 = torch.nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(self.embedding_dim)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        )

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):
        # Multi-Head Self-Attention
        inputs_norm = self.layer_norm1(inputs)
        Q = self.W_Q(inputs_norm)
        K = self.W_K(inputs_norm)
        V = self.W_V(inputs_norm)

        Q = Q.view(inputs.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(inputs.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(inputs.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)

        concat_output = head_outputs.transpose(1, 2).contiguous().view(inputs.size(0), -1, self.embedding_dim)
        attn_output = self.W_O(concat_output)
        attn_output = self.dropout(attn_output)
        out = inputs + attn_output

        # Feed-Forward Network
        out_norm = self.layer_norm2(out)
        ffn_output = self.ffn(out_norm)
        ffn_output = self.dropout(ffn_output)
        out = out + ffn_output

        return out

class ENCODER(torch.nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4, max_sequence_length=16):
        super(ENCODER, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.register_buffer("pos_encoding", get_positional_encoding(self.max_sequence_length, self.embedding_dim))

        self.magics = torch.nn.ModuleList([
            Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) 
            for _ in range(4)
        ])
    
    def forward(self, inputs):
        batch_size, group_size, embedding_dim = inputs.size()
        device = inputs.device

        pos_encodings = self.pos_encoding[:group_size, :].to(device)
        pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, group_size, embedding_dim]
        embs = inputs + pos_encodings

        for magic in self.magics:
            embs = magic(embs)
        return embs

# Define Image Embedding Model
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(196, embedding_dim)  # 14x14 patches flattened to 196

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Define Custom Dataset
def num2word(num: int):
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[num]

class MNISTSequenceDataset(Dataset):
    def __init__(self, indices, sequence_length=16, data_path='./data', train=True, vocabulary=None):
        self.indices = indices
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_dataset = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)
        self.num_sequences = len(self.indices) // (sequence_length // 4)  # 4 patches per image

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * (self.sequence_length // 4)  # Each sequence has 4 images
        patches = []
        labels = []

        for i in range(self.sequence_length // 4):
            index = self.indices[start_idx + i]
            image, label = self.mnist_dataset[index]

            # Convert label to word and get vocabulary index
            label_word = num2word(label)
            label_idx = self.vocabulary[label_word]

            # Divide the image into 4 patches
            patch_size = 14  # MNIST images are 28x28
            patches.append(image[:, :patch_size, :patch_size])    # Top-left
            patches.append(image[:, :patch_size, patch_size:])    # Top-right
            patches.append(image[:, patch_size:, :patch_size])    # Bottom-left
            patches.append(image[:, patch_size:, patch_size:])    # Bottom-right

            labels.extend([label_idx] * 4)  # Same label for all patches

        # Add <SOS> and <EOS> tokens
        labels = [self.vocabulary['<SOS>']] + labels + [self.vocabulary['<EOS>']]

        patches = torch.stack(patches)  # [sequence_length, 1, 14, 14]
        labels = torch.tensor(labels, dtype=torch.long)  # [sequence_length + 2]

        return patches, labels

# Define Transformer Model with Dual Decoders and GRL
class TransformerWithTwoDecoders(torch.nn.Module):
    def __init__(self, embedding_model, vocab_size, sequence_length, decoder_embedding_dim, encoder_embedding_dim, grl_lambda=1.0):
        super(TransformerWithTwoDecoders, self).__init__()
        self.embedding_model = embedding_model
        self.encoder = ENCODER(
            embedding_dim=encoder_embedding_dim,
            num_heads=2,
            max_sequence_length=sequence_length
        )
        self.assoc_decoder = ADECODER(
            vocab_size=vocab_size,
            decoder_embedding_dim=decoder_embedding_dim, 
            encoder_embedding_dim=encoder_embedding_dim,
            num_heads=2,
            max_sequence_length=sequence_length + 2  # +2 for <SOS> and <EOS>
        )
        self.dissoc_decoder = DDECODER(
            vocab_size=vocab_size,
            decoder_embedding_dim=decoder_embedding_dim, 
            encoder_embedding_dim=encoder_embedding_dim,
            num_heads=2,
            max_sequence_length=sequence_length + 2  # +2 for <SOS> and <EOS>
        )
        self.grl = GradientReversalLayer(lambda_=grl_lambda)  # Initialize GRL

    def forward(self, images, target_sequence):
        batch_size, sequence_length, channels, height, width = images.size()
        images = images.view(-1, channels, height, width)  # Flatten batch and sequence dimensions

        # Generate embeddings
        embeddings = self.embedding_model(images)  # [batch_size * sequence_length, embedding_dim]
        embeddings = embeddings.view(batch_size, sequence_length, -1)  # [batch_size, sequence_length, embedding_dim]

        # Pass through encoder
        encoder_output = self.encoder(embeddings)  # [batch_size, sequence_length, encoder_embedding_dim]

        # Decode with associative decoder
        assoc_output = self.assoc_decoder(target_sequence, encoder_output)  # [batch_size, target_seq_len, vocab_size]

        # Apply GRL before dissociative decoder
        grl_output = self.grl(encoder_output)  # [batch_size, sequence_length, encoder_embedding_dim]

        # Decode with dissociative decoder
        dissoc_output = self.dissoc_decoder(target_sequence, grl_output)  # [batch_size, target_seq_len, vocab_size]

        return assoc_output, dissoc_output

# Define Vocabulary
vocabulary = {
    '<SOS>': 0, 
    '<EOS>': 1, 
    'zero': 2, 
    'one': 3, 
    'two': 4, 
    'three': 5, 
    'four': 6, 
    'five': 7, 
    'six': 8, 
    'seven': 9, 
    'eight': 10, 
    'nine': 11
}
vocab_size = len(vocabulary)
print(f"Vocabulary size: {vocab_size}")

# Generate Random Indices
random_indices = random.sample(range(60000), 60000)  # Shuffle full MNIST dataset

# Define Dataset and DataLoader
sequence_length = 16
batch_size = 128
mnist_sequence_dataset = MNISTSequenceDataset(
    indices=random_indices,
    sequence_length=sequence_length,
    vocabulary=vocabulary
)
data_loader = DataLoader(
    mnist_sequence_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)

# Instantiate Models and Move to Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
decoder_embedding_dim = 64
encoder_embedding_dim = 128
img_embedding_dim = encoder_embedding_dim

embedding_model = ImageToEmbedding(embedding_dim=img_embedding_dim).to(device)
model = TransformerWithTwoDecoders(
    embedding_model, 
    vocab_size=vocab_size, 
    sequence_length=sequence_length, 
    decoder_embedding_dim=decoder_embedding_dim, 
    encoder_embedding_dim=encoder_embedding_dim,
    grl_lambda=1.0  # Start with lambda=1.0, adjust as needed
).to(device)

# Initialize Weights
def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# Define Loss Functions and Optimizer
assoc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
dissoc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training Loop
num_epochs = 100
alpha = 0.05  # Initial weight for dissociative loss

for epoch in range(num_epochs):
    alpha = 0.05 + (0.05 * epoch / num_epochs)

    model.train()  # Set model to training mode
    total_assoc_loss = 0.0
    total_dissoc_loss = 0.0
    total_loss_epoch = 0.0
    correct_assoc = 0
    total_assoc = 0

    for images, target_labels in data_loader:
        optimizer.zero_grad()

        images = images.to(device)
        target_labels = target_labels.to(device)

        input_sequences = target_labels[:, :-1]  # Remove <EOS>
        target_outputs = target_labels[:, 1:]    # Remove <SOS>

        # Forward pass
        assoc_output, dissoc_output = model(images, input_sequences)  # Each: [batch_size, seq_len, vocab_size]

        # Reshape for loss computation using .reshape()
        assoc_output_flat = assoc_output.reshape(-1, vocab_size)       # [batch_size * seq_len, vocab_size]
        dissoc_output_flat = dissoc_output.reshape(-1, vocab_size)     # [batch_size * seq_len, vocab_size]
        target_outputs_flat = target_outputs.reshape(-1)              # [batch_size * seq_len]

        # Compute associative loss (minimize)
        assoc_loss = assoc_criterion(assoc_output_flat, target_outputs_flat)

        # Compute dissociative loss (maximize)
        dissoc_loss = dissoc_criterion(dissoc_output_flat, target_outputs_flat)
        dissoc_loss = -dissoc_loss  # Maximize dissociative loss
        # Total loss
        total_loss = assoc_loss + alpha * dissoc_loss

        # Backward pass and optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # Accumulate losses for logging
        total_assoc_loss += assoc_loss.item()
        total_dissoc_loss += dissoc_loss.item()
        total_loss_epoch += total_loss.item()

        # Compute associative accuracy
        _, predicted = torch.max(assoc_output_flat, dim=1)
        correct_assoc += (predicted == target_outputs_flat).sum().item()
        total_assoc += target_outputs_flat.size(0)

    # Compute average losses and accuracy
    avg_assoc_loss = total_assoc_loss / len(data_loader)
    avg_dissoc_loss = total_dissoc_loss / len(data_loader)
    avg_total_loss = total_loss_epoch / len(data_loader)
    assoc_accuracy = correct_assoc / total_assoc

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Association Loss: {avg_assoc_loss:.4f}, "
          f"Dissociation Loss: {avg_dissoc_loss:.4f}, "
          f"Total Loss: {avg_total_loss:.4f}, "
          f"Assoc Accuracy: {assoc_accuracy:.4f}")

    # Save checkpoints every 10 epochs
    if (epoch + 1) % 10 == 0:
        model_filename = f"./models/model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Checkpoint saved as {model_filename}")

# Save the final model
model_filename = f"./models/model_bidec_image_split_one_go_embedding{img_embedding_dim}_decoder{decoder_embedding_dim}_encoder{encoder_embedding_dim}_seq{sequence_length}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}_ls{avg_total_loss:.2f}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")
