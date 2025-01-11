import torch
import json
import random
from encoder import ENCODER as Encoder  # Import the ENCODER class
from associative_decoder import ADECODER   # Import the DECODER class
from dissociative_decoder import DDECODER # Import the DECODER classcoder import ADECODER as ADecoder  # Import the DECODER class
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(29)

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image-to-embedding model for MNIST patches
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()  # Flatten 14x14 into 196
        self.linear = torch.nn.Linear(196, embedding_dim)  # Map 196 to embedding_dim

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input patch
        x = self.linear(x)   # Linear layer to create 64-dim embedding
        return x

def num2word(num: int):
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[num]

# Custom dataset returning image patches and labels
class MNISTSequenceDataset(Dataset):
    def __init__(self, indices, sequence_length=16, data_path='./data', train=True, vocabulary=None):
        self.indices = indices
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_dataset = datasets.MNIST(root=data_path, train=train, download=True, transform=transform)
        self.num_sequences = len(self.indices) // (sequence_length // 4)  # Adjust for 4 patches per image

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * (self.sequence_length // 4)  # Each sequence consists of 4 images
        patches = []
        labels = []

        for i in range(self.sequence_length // 4):
            index = self.indices[start_idx + i]
            image, label = self.mnist_dataset[index]

            # Convert label (integer) to word and get vocabulary index
            label_word = num2word(label)
            label_idx = self.vocabulary[label_word]

            # Divide the image into 4 patches
            patch_size = 14  # Half of the 28x28 MNIST image
            patches.append(image[:, :patch_size, :patch_size])    # Top-left
            patches.append(image[:, :patch_size, patch_size:])    # Top-right
            patches.append(image[:, patch_size:, :patch_size])    # Bottom-left
            patches.append(image[:, patch_size:, patch_size:])    # Bottom-right

            labels.extend([label_idx] * 4)  # Each patch is associated with the same label

        # Add <SOS> and <EOS> tokens to labels
        labels = [self.vocabulary['<SOS>']] + labels + [self.vocabulary['<EOS>']]
        
        patches = torch.stack(patches)  # Shape: [sequence_length, 1, 14, 14]
        labels = torch.tensor(labels, dtype=torch.long)  # Shape: [sequence_length + 2]

        return patches, labels

# Transformer model with embedding model, encoder, and decoder
class TransformerWithTwoDecoders(torch.nn.Module):
    def __init__(self, embedding_model, vocab_size, sequence_length, decoder_embedding_dim, encoder_embedding_dim):
        super(TransformerWithTwoDecoders, self).__init__()
        self.embedding_model = embedding_model
        self.encoder = Encoder(
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

    def forward(self, images, target_sequence):
        batch_size, sequence_length, channels, height, width = images.size()
        images = images.view(-1, channels, height, width)  # Flatten batch and sequence dimensions

        # Generate embeddings
        embeddings = self.embedding_model(images)  # Shape: [batch_size * sequence_length, embedding_dim]
        embeddings = embeddings.view(batch_size, sequence_length, -1)  # Reshape back to [batch_size, sequence_length, embedding_dim]

        # Pass through encoder
        encoder_output = self.encoder(embeddings)  # Shape: [batch_size, sequence_length, encoder_embedding_dim]

        # Decode with associative decoder
        assoc_output = self.assoc_decoder(target_sequence, encoder_output)  # Shape: [batch_size, target_seq_len, vocab_size]

        # Decode with dissociative decoder
        dissoc_output = self.dissoc_decoder(target_sequence, encoder_output)  # Shape: [batch_size, target_seq_len, vocab_size]

        return assoc_output, dissoc_output


##########################################################################################

# Generate random indices from 0 to 60000
random_indices = random.sample(range(60000), 60000)  # Use full MNIST dataset

sequence_length = 16
batch_size = 128  # Adjust as needed based on your GPU memory
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

# Instantiate dataset and data loader
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

decoder_embedding_dim = 32
encoder_embedding_dim = 64

img_embedding_dim = encoder_embedding_dim
# Instantiate embedding model and move to device
embedding_model = ImageToEmbedding(embedding_dim=img_embedding_dim).to(device)
# Instantiate the Transformer model and move to device
model = TransformerWithTwoDecoders(embedding_model, vocab_size, sequence_length, decoder_embedding_dim=decoder_embedding_dim, encoder_embedding_dim=encoder_embedding_dim).to(device)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# Define loss function and optimizer
assoc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
dissoc_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # You may customize this
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100
alpha = 0.5  # Weight for dissociative loss; adjust as needed

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_assoc_loss = 0.0
    total_dissoc_loss = 0.0
    total_loss_epoch = 0.0

    for images, target_labels in data_loader:
        optimizer.zero_grad()

        images = images.to(device)
        target_labels = target_labels.to(device)

        input_sequences = target_labels[:, :-1]  # Remove <EOS>
        target_outputs = target_labels[:, 1:]    # Remove <SOS>

        # Forward pass
        assoc_output, dissoc_output = model(images, input_sequences)  # Each: [batch_size, seq_len, vocab_size]

        # Reshape for loss computation
        assoc_output = assoc_output.view(-1, vocab_size)       # [batch_size * seq_len, vocab_size]
        dissoc_output = dissoc_output.view(-1, vocab_size)     # [batch_size * seq_len, vocab_size]
        target_outputs = target_outputs.reshape(-1)            # [batch_size * seq_len]

        # Compute associative loss (minimize)
        assoc_loss = assoc_criterion(assoc_output, target_outputs)
        with torch.no_grad():
            _, predicted = torch.max(assoc_output, dim=1)
            correct = (predicted == target_outputs).sum().item()
            total = target_outputs.size(0)
            assoc_accuracy = correct / total

        # Compute dissociative loss (maximize)
        # Option 1: Flip the dissociative loss
        dissoc_loss = -dissoc_criterion(dissoc_output, target_outputs)
        with torch.no_grad():
            _, predicted = torch.max(dissoc_output, dim=1)
            correct = (predicted == target_outputs).sum().item()
            total = target_outputs.size(0)
            dissoc_accuracy = correct / total

        # Option 2: Use a separate strategy, e.g., training dissoc decoder on shuffled labels
        # Uncomment below if using shuffled labels
        """
        shuffled_labels = target_labels[:, 1:].clone()
        shuffled_labels = shuffled_labels.view(-1)
        shuffled_indices = torch.randperm(shuffled_labels.size(0))
        shuffled_labels = shuffled_labels[shuffled_indices]

        dissoc_loss = dissoc_criterion(dissoc_output, shuffled_labels)
        """

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

    # Compute average losses
    avg_assoc_loss = total_assoc_loss / len(data_loader)
    avg_dissoc_loss = total_dissoc_loss / len(data_loader)
    avg_total_loss = total_loss_epoch / len(data_loader)
    print("Assoc accuracy:", assoc_accuracy)
    print("Dissoc accuracy:", dissoc_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Association Loss: {avg_assoc_loss:.4f}, Dissociation Loss: {avg_dissoc_loss:.4f}, Total Loss: {avg_total_loss:.4f}")


# Save the model
model_filename = f"./models/model_image_split_embedding{img_embedding_dim}_decoder{decoder_embedding_dim}_encoder{encoder_embedding_dim}_seq{sequence_length}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}_ls{avg_total_loss:.2f}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")
