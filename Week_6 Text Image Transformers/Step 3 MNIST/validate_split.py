import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from encoder_split import ENCODER as Encoder  # Ensure correct import paths
from decoder_split import DECODER as Decoder  # Ensure correct import paths

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define vocabulary (matching the model requirements)
vocabulary = {word: idx for idx, word in enumerate(['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])}
inv_vocabulary = {idx: word for word, idx in vocabulary.items()}  # For mapping indices back to words

vocab_size = len(vocabulary)

# Function to convert number to word
def num2word(num: int):
    words = ['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return words[num]

# Custom dataset returning image patches and labels
class MNISTSequenceDataset(Dataset):
    def __init__(self, indices, sequence_length=16, data_path='./data', train=True, vocabulary=None):
        self.indices = indices
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_dataset = datasets.MNIST(root=data_path, train=train, download=False, transform=transform)
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

# Instantiate the test dataset and data loader using MNISTSequenceDataset
test_indices = list(range(10000))  # MNIST test set has 10,000 images
sequence_length = 16

mnist_test_sequence_dataset = MNISTSequenceDataset(
    indices=test_indices,
    sequence_length=sequence_length,
    train=False,  # Indicate that this is test data
    vocabulary=vocabulary
)

test_loader = DataLoader(
    mnist_test_sequence_dataset,
    batch_size=64,  # Adjust as needed based on your GPU memory
    shuffle=False,
    drop_last=False
)

# Define the corrected image-to-embedding model
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()  # Flatten 14x14 into 196
        self.linear = torch.nn.Linear(196, embedding_dim)  # Corrected: 196 -> embedding_dim

    def forward(self, x):
        x = self.flatten(x)  # Shape: [batch_size * sequence_length, 196]
        x = self.linear(x)   # Shape: [batch_size * sequence_length, embedding_dim]
        return x

# Define the Transformer model (ensure it matches the architecture used during training)
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
            num_heads=2,  # Ensure this matches training
            max_sequence_length=(sequence_length//4) + 2  # +2 for <SOS> and <EOS>
        )

    def forward(self, images, target_sequence):
        # images shape: [batch_size, sequence_length, 1, 14, 14]
        batch_size, sequence_length, channels, height, width = images.size()
        images = images.view(-1, channels, height, width)  # Flatten batch and sequence dimensions

        # Generate embeddings
        embeddings = self.embedding_model(images)  # Shape: [batch_size * sequence_length, embedding_dim]
        embeddings = embeddings.view(batch_size, sequence_length, -1)  # Reshape back to [batch_size, sequence_length, embedding_dim]

        # Pass through encoder and decoder
        encoder_output = self.encoder(embeddings)
        decoder_output = self.decoder(target_sequence, encoder_output)
        return decoder_output

# Instantiate embedding model and Transformer model
decoder_embedding_dim = 32
encoder_embedding_dim = 128
sequence_length = 16

embedding_model = ImageToEmbedding(embedding_dim=encoder_embedding_dim).to(device)
model = Transformer(
    embedding_model, 
    vocab_size, 
    sequence_length, 
    decoder_embedding_dim=decoder_embedding_dim, 
    encoder_embedding_dim=encoder_embedding_dim
).to(device)

# Load the model's state dictionary correctly
model_path = './models/model_image_split_embedding128_decoder32_encoder128_seq16_batch128_epochs30_lr0.0001_ls0.74.pth'  # Replace with your model file path
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Updated Test function with Autoregressive Decoding
def test(model, test_loader, vocabulary, device, max_length=6):
    model.eval()
    total_correct = 0
    total_tokens = 0
    sos_token = vocabulary['<SOS>']
    eos_token = vocabulary['<EOS>']

    # Create an inverse vocabulary to map indices back to words (optional, for interpretation)
    inv_vocabulary = {idx: word for word, idx in vocabulary.items()}

    with torch.no_grad():
        for batch_idx, (patches, labels) in enumerate(test_loader):
            patches = patches.to(device)  # Shape: [batch_size, 16, 1, 14, 14]
            labels = labels.to(device)    # Shape: [batch_size, 6] -> [<SOS>, four, six, two, five, <EOS>]

            batch_size = patches.size(0)

            # Initialize the input sequence with <SOS> token
            generated_sequences = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)  # [batch_size, 1]

            # Store the generated tokens
            generated_tokens = [[] for _ in range(batch_size)]

            for _ in range(max_length -1):  # Already have <SOS>, generate up to <EOS>
                # Forward pass
                output = model(patches, generated_sequences)  # Output shape: [batch_size, current_length, vocab_size]

                # Get the last token's logits
                last_token_logits = output[:, -1, :]  # [batch_size, vocab_size]

                # Get the predicted tokens (greedy decoding)
                predicted_tokens = last_token_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]

                # Append the predicted tokens to the generated sequences
                generated_sequences = torch.cat((generated_sequences, predicted_tokens), dim=1)  # [batch_size, current_length +1]

                # Collect the generated tokens
                for i in range(batch_size):
                    token = predicted_tokens[i].item()
                    generated_tokens[i].append(token)

                # Check if all sequences have generated <EOS>
                if (predicted_tokens == eos_token).all():
                    break

            # Compare generated tokens with target labels
            # Extract target labels by removing <SOS> and <EOS>
            target_labels = labels[:, 1:-1]  # [batch_size, 4]

            # Convert generated tokens to tensor
            # Initialize with <EOS> for padding
            generated_tokens_tensor = torch.full((batch_size, target_labels.size(1)), eos_token, dtype=torch.long, device=device)

            for i in range(batch_size):
                seq = generated_tokens[i]
                # Truncate or pad the sequence to match the target length
                for j in range(min(len(seq), target_labels.size(1))):
                    generated_tokens_tensor[i, j] = seq[j]

            # Compare token by token
            correct = (generated_tokens_tensor == target_labels).sum().item()
            total_correct += correct
            total_tokens += target_labels.numel()

            # Optional: Print some examples for the first batch
            if batch_idx == 0:
                for i in range(min(5, batch_size)):
                    generated_sequence = [inv_vocabulary.get(token, "<UNK>") for token in generated_tokens[i]]
                    target_sequence = [inv_vocabulary.get(token.item(), "<UNK>") for token in target_labels[i]]
                    print(f"Generated: {' '.join(generated_sequence)}")
                    print(f"Target: {' '.join(target_sequence)}\n")

    accuracy = total_correct / total_tokens
    print(f"Test Accuracy: {accuracy:.4f}")


# Run the test
test(model, test_loader, vocabulary, device)
