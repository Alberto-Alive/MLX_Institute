import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from encoder import ENCODER as Encoder  # Import the ENCODER class
from decoder import DECODER as Decoder  # Import the DECODER class

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your vocabulary if needed (matching the model requirements)
vocabulary = {word: idx for idx, word in enumerate(['<SOS>', '<EOS>', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'])}

# Check the vocabulary
print("Vocabulary:", vocabulary)
vocab_size = len(vocabulary)

# Load the MNIST test dataset
test_transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define your model (should match the architecture of the saved model)
class ImageToEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(784, embedding_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x



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
            num_heads=4,
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

# Load the Transformer model (define architecture as per the saved file)
# embedding_dim = 128
decoder_embedding_dim = 16
encoder_embedding_dim = 128
sequence_length = 16

embedding_model = ImageToEmbedding(encoder_embedding_dim).to(device)
model = Transformer(embedding_model, vocab_size, sequence_length, decoder_embedding_dim=decoder_embedding_dim, encoder_embedding_dim=encoder_embedding_dim).to(device)

# Load the model's state dictionary
model_path = './models/model_image_embedding128_decoder16_encoder128_seq16_batch128_epochs10_lr0.0005_ls2.16.pth'  # Replace with your model file path
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Test function to evaluate the model on the MNIST test dataset
def test(model, test_loader, vocabulary, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    sos_token = vocabulary['<SOS>']
    eos_token = vocabulary['<EOS>']

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Prepare images and labels on device
            images = images.to(device)
            labels = labels.to(device)

            # Reshape images to add a channel dimension
            images = images.unsqueeze(1)  # Shape: [batch_size, 1, 28, 28]

            # Correctly construct input_sequences with a single <SOS> token and the label
            batch_size = labels.size(0)
            input_sequences = torch.cat([
                torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device),  # <SOS> token
                labels.unsqueeze(1)  # Label itself
            ], dim=1)  # Final shape: [batch_size, 2]

            # Correctly construct target_outputs with the label and <EOS>
            target_outputs = torch.cat([
                labels.unsqueeze(1),  # Label itself
                torch.full((batch_size, 1), eos_token, dtype=torch.long, device=device)  # <EOS> token
            ], dim=1)  # Final shape: [batch_size, 2]

            # Debug: Print input_sequences and target_outputs for the first batch
            if batch_idx == 0:
                print("Corrected input_sequences:", input_sequences[0])  # Should be [<SOS>, label]
                print("Corrected target_outputs:", target_outputs[0])    # Should be [label, <EOS>]

            # Forward pass through the model
            output = model(images, input_sequences)  # Output shape: [batch_size, sequence_length + 1, vocab_size]

            # Get predictions and match to target length
            predictions = output.argmax(dim=-1)  # Shape: [batch_size, sequence_length + 1]
            predictions = predictions[:, :target_outputs.size(1)]  # Ensure predictions and target_outputs are the same length

            # Calculate accuracy
            correct = (predictions == target_outputs).sum().item()
            total_correct += correct
            total_tokens += target_outputs.numel()

    accuracy = total_correct / total_tokens
    print(f"Test Accuracy: {accuracy:.4f}")


# Run the test
test(model, test_loader, vocabulary, device)
