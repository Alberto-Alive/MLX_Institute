import torch
import json
import random
from encoder import ENCODER as Encoder  # Import the ENCODER class
from decoder import DECODER as Decoder  # Import the DECODER class
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(29)

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image-to-embedding model for MNIST images
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()  # Flatten 28x28 into 784
        self.linear = torch.nn.Linear(784, embedding_dim)  # Map 784 to 64

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.linear(x)   # Linear layer to create 64-dim embedding
        return x

def num2word(num: int):
    words = ['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[num]

# Custom dataset returning images and labels
class MNISTSequenceDataset(Dataset):
    def __init__(
        self,
        indices,
        sequence_length=16,
        data_path='./data',
        train=True,
        vocabulary=None
    ):
        self.indices = indices
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary  # Assign the passed vocabulary

        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_dataset = datasets.MNIST(
            root=data_path, train=train, download=True, transform=transform
        )
        self.num_sequences = len(self.indices) // self.sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        images = []
        labels = []

        for i in range(self.sequence_length):
            index = self.indices[start_idx + i]
            image, label = self.mnist_dataset[index]
            images.append(image)

            # Convert label (integer) to word
            label_word = num2word(label)
            # Get index of word from vocabulary
            label_idx = self.vocabulary[label_word]
            labels.append(label_idx)

        # Add <SOS> and <EOS> tokens to labels
        labels = [self.vocabulary['<SOS>']] + labels + [self.vocabulary['<EOS>']]

        images = torch.stack(images)  # Shape: [sequence_length, 1, 28, 28]
        labels = torch.tensor(labels, dtype=torch.long)  # Shape: [sequence_length + 2]

        return images, labels

# Transformer model with embedding model, encoder, and decoder
class Transformer(torch.nn.Module):
    def __init__(self, embedding_model, vocab_size, sequence_length, decoder_embedding_dim, encoder_embedding_dim):
        super(Transformer, self).__init__()
        self.embedding_model = embedding_model
        self.encoder = Encoder(
            embedding_dim=encoder_embedding_dim,
            num_heads=4,
            max_sequence_length=sequence_length
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            decoder_embedding_dim=decoder_embedding_dim, 
            encoder_embedding_dim=encoder_embedding_dim,
            num_heads=2,
            max_sequence_length=sequence_length + 2  # +2 for <SOS> and <EOS>
        )

    def forward(self, images, target_sequence):
        # images shape: [batch_size, sequence_length, 1, 28, 28]
        batch_size, sequence_length, channels, height, width = images.size()
        images = images.view(-1, channels, height, width)  # Flatten batch and sequence dimensions

        # Generate embeddings
        embeddings = self.embedding_model(images)  # Shape: [batch_size * sequence_length, embedding_dim]
        embeddings = embeddings.view(batch_size, sequence_length, -1)  # Reshape back to [batch_size, sequence_length, embedding_dim]

        # Pass through encoder and decoder
        encoder_output = self.encoder(embeddings)
        decoder_output = self.decoder(target_sequence, encoder_output)
        return decoder_output

##########################################################################################

# Generate random indices from 0 to 60000
random_indices = random.sample(range(60000), 60000)  # Use full MNIST dataset

sequence_length = 16
batch_size = 128  # Adjust as needed based on your GPU memory
vocabulary = {word: idx for idx, word in enumerate(['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])}

# Check the vocabulary
print("Vocabulary:", vocabulary)
vocab_size = len(vocabulary)

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

decoder_embedding_dim = 16
encoder_embedding_dim = 128


img_embedding_dim = encoder_embedding_dim
# Instantiate embedding model and move to device
embedding_model = ImageToEmbedding(embedding_dim=img_embedding_dim).to(device)
# Instantiate the Transformer model and move to device
model = Transformer(embedding_model, vocab_size, sequence_length, decoder_embedding_dim=decoder_embedding_dim, encoder_embedding_dim=encoder_embedding_dim).to(device)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(initialize_weights)


# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    for images, target_labels in data_loader:
        optimizer.zero_grad()

        images = images.to(device)
        target_labels = target_labels.to(device)

        input_sequences = target_labels[:, :-1]
        target_outputs = target_labels[:, 1:]

        # Forward pass
        output = model(images, input_sequences)  # Output shape: [batch_size, sequence_length + 1, vocab_size]

        # Compute loss
        loss = criterion(output.view(-1, vocab_size), target_outputs.reshape(-1))

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


model_filename = f"./models/model_image_embedding{img_embedding_dim}_decoder{decoder_embedding_dim}_encoder{encoder_embedding_dim}_seq{sequence_length}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}_ls{loss.item():.2f}.pth"

# Save the model
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")