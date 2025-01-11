import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sentencepiece as spm
import ast

# Import the updated ENCODER and DECODER classes
from encoder_split import ENCODER as Encoder  # Now includes embedding
from decoder_split import DECODER as Decoder  # Ensure this matches your decoder implementation

torch.manual_seed(29)

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device set to:", device)

##########################################################################################
# Dataset Class for Flickr30k
##########################################################################################

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, tokenizer=None, max_seq_length=50, split='train'):
        """
        Args:
            image_dir (str): Directory with all the images.
            captions_file (str): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on an image.
            tokenizer (callable): Tokenizer to process captions.
            max_seq_length (int): Maximum sequence length for captions.
            split (str): 'train', 'val', or 'test' to select the data split.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split.lower()

        # Read the CSV
        try:
            self.captions_df = pd.read_csv(captions_file)
        except Exception as e:
            raise ValueError(f"Error reading the captions CSV file: {e}")

        # Filter by split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'.")
        self.captions_df = self.captions_df[self.captions_df['split'] == self.split]

        # Initialize list for (image_path, caption) pairs
        self.image_caption_pairs = []

        for _, row in self.captions_df.iterrows():
            image_id = row['filename']  # Adjust based on your CSV column
            captions_str = row['raw']    # Assuming 'raw' contains the list of captions
            try:
                # Replace double double-quotes with single double-quotes
                captions_str = captions_str.replace('""', '"').replace('"', '').strip()
                # Handle cases where captions are enclosed in brackets
                if captions_str.startswith('[') and captions_str.endswith(']'):
                    captions_str = captions_str[1:-1]
                # Split captions by '","' or similar separators
                captions = [caption.strip() for caption in captions_str.split('","')]

                for caption in captions:
                    # Remove any residual quotes
                    caption = caption.strip('"').strip()
                    image_path = os.path.join(self.image_dir, image_id)
                    if os.path.exists(image_path):
                        self.image_caption_pairs.append((image_path, caption))
                    else:
                        print(f"Warning: Image {image_path} does not exist.")
            except Exception as e:
                print(f"Error parsing captions for image {image_id}: {e}")

        if not self.image_caption_pairs:
            raise ValueError(f"No (image, caption) pairs found for split '{self.split}'.")

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_path, caption = self.image_caption_pairs[idx]

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

        if self.transform:
            image = self.transform(image)  # Apply transformations

        # Split image into 8x8 patches (28x28 grid, total 784 patches)
        patch_size = 8
        # image size: [C, H, W] = [3, 224, 224]
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # patches shape: [C, 28, 28, 8, 8]
        patches = patches.contiguous().view(3, -1, patch_size, patch_size)  # [3, 784, 8, 8]
        patches = patches.permute(1, 0, 2, 3)  # [784, 3, 8, 8]

        # Tokenize the caption
        tokens = self.tokenizer.encode(caption)
        # Add <SOS> and <EOS> tokens
        sos_id = self.tokenizer.piece_to_id('<s>')
        eos_id = self.tokenizer.piece_to_id('</s>')
        tokens = [sos_id] + tokens + [eos_id]

        # Truncate or pad tokens to max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            tokens[-1] = eos_id
        else:
            tokens += [self.tokenizer.pad_id()] * (self.max_seq_length - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        return patches, tokens

##########################################################################################
# Transformer Model
##########################################################################################

class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, max_seq_length):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

    def forward(self, images, target_sequence):
        """
        images: [batch_size, 784, 3, 8, 8]  # 784 patches per image
        target_sequence: [batch_size, max_seq_length]  # <SOS> + tokens + <EOS>
        """
        # Pass through the encoder
        encoder_output = self.encoder(images)  # [batch_size, 784, embedding_dim]

        # Pass through the decoder
        decoder_output = self.decoder(target_sequence, encoder_output)  # [batch_size, max_seq_length, vocab_size]

        return decoder_output

##########################################################################################
# Initialization and Training
##########################################################################################

def main():
    # Paths
    image_dir = './data/flickr30k-images-resized'
    captions_file = './data/flicker_annotations_30k.csv'
    tokenizer_path = './models/caption_tokenizer.model'

    # Load the tokenizer
    tokenizer = spm.SentencePieceProcessor()
    try:
        tokenizer.load(tokenizer_path)
    except Exception as e:
        raise ValueError(f"Error loading tokenizer model from {tokenizer_path}: {e}")
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary Size: {vocab_size}")

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Removed transforms.Resize since images are already resized
        # Optional: Add data augmentation here if desired
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
    ])

    # Instantiate dataset for training
    flickr_dataset = Flickr30kDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        tokenizer=tokenizer,
        max_seq_length=50,  # Adjust as needed
        split='train'       # Specify 'train', 'val', or 'test'
    )

    # DataLoader parameters
    batch_size = 50  # As per your requirement

    data_loader = DataLoader(
        flickr_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Ensure all batches are full
    )

    # Model parameters
    decoder_embedding_dim = 256  # Adjust as needed
    encoder_embedding_dim = 128  # Should match the encoder's embedding_dim
    num_patches = 784  # 28x28 grid
    max_seq_length = 50  # As defined in dataset

    # Instantiate encoder and decoder, and move to device
    encoder = Encoder(
        embedding_dim=encoder_embedding_dim,
        num_heads=8,  # Increased number of heads to handle larger input
        max_sequence_length=num_patches
    ).to(device)

    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_embedding_dim=decoder_embedding_dim, 
        encoder_embedding_dim=encoder_embedding_dim,
        num_heads=4,  # Adjusted as needed
        max_sequence_length=max_seq_length  # For captions
    ).to(device)

    # Instantiate the Transformer model and move to device
    model = TransformerModel(
        encoder, 
        decoder, 
        vocab_size, 
        max_seq_length
    ).to(device)

    # Initialize weights
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(initialize_weights)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        total_loss = 0
        for batch_idx, (images, target_labels) in enumerate(data_loader):
            optimizer.zero_grad()

            images = images.to(device)          # [batch_size, 784, 3, 8, 8]
            target_labels = target_labels.to(device)  # [batch_size, max_seq_length]

            input_sequences = target_labels[:, :-1]  # [batch_size, max_seq_length -1]
            target_outputs = target_labels[:, 1:]    # [batch_size, max_seq_length -1]

            # Forward pass
            output = model(images, input_sequences)  # [batch_size, max_seq_length -1, vocab_size]

            # Compute loss
            output = output.view(-1, vocab_size)  # [(batch_size * (max_seq_length -1)), vocab_size]
            target_outputs = target_outputs.view(-1)  # [(batch_size * (max_seq_length -1))]
            loss = criterion(output, target_outputs)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    model_filename = (
        f"./models/model_flickr30k_decoder{decoder_embedding_dim}_"
        f"encoder{encoder_embedding_dim}_seq{num_patches}_batch{batch_size}_"
        f"epochs{num_epochs}_lr{learning_rate}_loss{avg_loss:.4f}.pth"
    )
    os.makedirs('./models', exist_ok=True)  # Ensure the models directory exists
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    main()
