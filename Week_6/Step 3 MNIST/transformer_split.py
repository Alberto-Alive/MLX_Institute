import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random

# Assuming ENCODER and DECODER classes are correctly imported from encoder_split and decoder_split
from encoder_split import ENCODER as Encoder  # Ensure this matches your encoder implementation
from decoder_split import DECODER as Decoder  # Ensure this matches your decoder implementation

torch.manual_seed(29)

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image-to-embedding model for MNIST patches
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()  # Flatten 14x14 into 196
        self.linear = torch.nn.Linear(196, embedding_dim)  # Map 196 to embedding_dim

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input patch
        x = self.linear(x)   # Linear layer to create embedding
        return x

def num2word(num: int):
    words = ['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[num]

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
        sequence_labels = []

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

            # Collect labels for the sequence
            sequence_labels.append(label_idx)

        # Define the sequence label as the sequence of image labels
        # Example: [<SOS>, four, six, two, five, <EOS>]
        labels = [self.vocabulary['<SOS>']] + sequence_labels + [self.vocabulary['<EOS>']]

        patches = torch.stack(patches)  # Shape: [16, 1, 14, 14]
        labels = torch.tensor(labels, dtype=torch.long)  # Shape: [6]

        return patches, labels

# Transformer model with embedding model, encoder, and decoder
class TransformerModel(torch.nn.Module):
    def __init__(self, embedding_model, vocab_size, sequence_length, decoder_embedding_dim, encoder_embedding_dim):
        super(TransformerModel, self).__init__()
        self.embedding_model = embedding_model
        self.encoder = Encoder(
            embedding_dim=encoder_embedding_dim,
            num_heads=4,
            max_sequence_length=(sequence_length // 4)  # Number of images per sequence
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            decoder_embedding_dim=decoder_embedding_dim, 
            encoder_embedding_dim=encoder_embedding_dim,
            num_heads=2,
            max_sequence_length=(sequence_length // 4) + 2  # +2 for <SOS> and <EOS>
        )

    def forward(self, images, target_sequence):
        """
        images: [batch_size, 16, 1, 14, 14]  # sequence_length=16 patches
        target_sequence: [batch_size, 6]  # <SOS> + 4 labels + <EOS>
        """
        batch_size, sequence_length, channels, height, width = images.size()
        
        # Step 1: Flatten batch and sequence dimensions to process all patches
        images = images.view(batch_size * sequence_length, channels, height, width)  # [batch_size*16, 1, 14, 14]

        # Step 2: Generate embeddings for each patch
        patch_embeddings = self.embedding_model(images)  # [batch_size*16, embedding_dim]

        # Step 3: Reshape to [batch_size, 4 images, 4 patches per image, embedding_dim]
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length // 4, 4, -1)  # [batch_size, 4, 4, embedding_dim]

        # Step 4: Aggregate patches per image (average pooling)
        image_embeddings = patch_embeddings.mean(dim=2)  # [batch_size, 4, embedding_dim]

        # Step 5: Pass through the encoder
        encoder_output = self.encoder(image_embeddings)  # [batch_size, 4, encoder_embedding_dim]

        # Step 6: Pass through the decoder
        decoder_output = self.decoder(target_sequence, encoder_output)  # [batch_size, 6, vocab_size]

        return decoder_output

##########################################################################################

# Generate random indices from 0 to 60000 (MNIST has 60,000 training samples)
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

# Check the vocabulary
print("Vocabulary:", vocabulary)
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
encoder_embedding_dim = 128

# Instantiate embedding model and move to device
embedding_model = ImageToEmbedding(embedding_dim=encoder_embedding_dim).to(device)

# Instantiate the Transformer model and move to device
model = TransformerModel(
    embedding_model, 
    vocab_size, 
    sequence_length, 
    decoder_embedding_dim=decoder_embedding_dim, 
    encoder_embedding_dim=encoder_embedding_dim
).to(device)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(initialize_weights)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    total_loss = 0
    for images, target_labels in data_loader:
        optimizer.zero_grad()

        images = images.to(device)          # [batch_size, 16, 1, 14, 14]
        target_labels = target_labels.to(device)  # [batch_size, 6]

        input_sequences = target_labels[:, :-1]  # [batch_size, 5] -> <SOS> + 4 labels
        target_outputs = target_labels[:, 1:]    # [batch_size, 5] -> 4 labels + <EOS>

        # Forward pass
        output = model(images, input_sequences)  # [batch_size, 5, vocab_size]

        # Compute loss
        loss = criterion(output.view(-1, vocab_size), target_outputs.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the model
model_filename = f"./models/model_image_split_embedding{encoder_embedding_dim}_decoder{decoder_embedding_dim}_encoder{encoder_embedding_dim}_seq{sequence_length}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}_ls{loss.item():.2f}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")
