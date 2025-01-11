# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from encoder import ENCODER
from decoder import DECODER
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu

# Configuration
DATA_DIR = "./preprocessed_dataset"  # Directory containing 'mel_*.npy' and 'tokens_*.pt' files
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MELS = 80  # Number of Mel bins in your spectrograms
NUM_HEADS = 4
ENCODER_EMBEDDING_DIM = 512
DECODER_EMBEDDING_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
MAX_SEQUENCE_LENGTH = 5000  # Adjust based on your data

# Special tokens as per the multitask format described in the paper
SPECIAL_TOKENS = {
    'bos_token': '<|startoftranscript|>',
    'eos_token': '<|endoftranscript|>',
    'additional_special_tokens': [
        '<|transcribe|>',
        '<|translate|>',
        '<|nospeech|>',
        '<|notimestamps|>',
        # Add any other special tokens needed
    ]
}

# Load the tokenizer
def load_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('tokenizer/')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

tokenizer = load_tokenizer()
print(f"Padding token ID: {tokenizer.pad_token_id}") 
VOCAB_SIZE = len(tokenizer)  # Includes added special tokens

# Custom Dataset
class SpeechToTextDataset(Dataset):
    def __init__(self, data_dir):
        super(SpeechToTextDataset, self).__init__()
        self.data_dir = data_dir
        # List all mel and token files
        self.mel_files = sorted([f for f in os.listdir(data_dir) if f.startswith("mel_")])
        self.token_files = sorted([f for f in os.listdir(data_dir) if f.startswith("tokens_")])

        # Ensure that mel and token files are matched
        assert len(self.mel_files) == len(self.token_files), "Mismatch between mel and token files"

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        # Load mel spectrogram
        mel_path = os.path.join(self.data_dir, self.mel_files[idx])
        mel = np.load(mel_path)  # Shape: (n_mels, time_steps)
        mel = torch.tensor(mel, dtype=torch.float32)

        # Load tokenized transcription
        tokens_path = os.path.join(self.data_dir, self.token_files[idx])
        tokens = torch.load(tokens_path, weights_only=True)  # Shape: (sequence_length,)

        return mel, tokens

# Collate function for DataLoader
def collate_fn(batch):
    """
    Pads sequences in the batch to the maximum length in the batch.
    """
    mels, tokens = zip(*batch)

    # Pad mels
    mel_lengths = [mel.shape[1] for mel in mels]
    max_mel_length = max(mel_lengths)
    padded_mels = []
    for mel in mels:
        pad_length = max_mel_length - mel.shape[1]
        if pad_length > 0:
            mel = torch.cat([mel, torch.zeros((N_MELS, pad_length))], dim=1)
        padded_mels.append(mel.unsqueeze(0))  # Add batch dimension

    mels_tensor = torch.cat(padded_mels, dim=0)  # Shape: (batch_size, n_mels, max_mel_length)

    # Pad tokens
    token_lengths = [len(t) for t in tokens]
    max_token_length = max(token_lengths)
    padded_tokens = []
    for t in tokens:
        pad_length = max_token_length - len(t)
        if pad_length > 0:
            t = torch.cat([t, torch.full((pad_length,), fill_value=tokenizer.pad_token_id, dtype=torch.long)])  # Padding token
        padded_tokens.append(t.unsqueeze(0))

    tokens_tensor = torch.cat(padded_tokens, dim=0)  # Shape: (batch_size, max_token_length)

    return mels_tensor, tokens_tensor, torch.tensor(mel_lengths), torch.tensor(token_lengths)

# Define truncate_at_eos function
def truncate_at_eos(tokens, eos_id):
    if eos_id in tokens:
        idx = tokens.index(eos_id)
        return tokens[:idx]
    else:
        return tokens

# Training Loop
def train():
    # Prepare dataset and dataloader
    dataset = SpeechToTextDataset(DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Adjust based on your system
        drop_last=True,
        pin_memory=True
    )

    # Initialize models
    encoder = ENCODER(
        n_mels=N_MELS,
        hidden_dim=ENCODER_EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        encoder_layers=NUM_ENCODER_LAYERS,
        num_heads=NUM_HEADS
    ).to(DEVICE)

    decoder = DECODER(
        vocab_size=VOCAB_SIZE,
        decoder_embedding_dim=DECODER_EMBEDDING_DIM,
        encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_layers=NUM_DECODER_LAYERS
    ).to(DEVICE)

    # Adjust decoder's embeddings to match the tokenizer's vocabulary size
    decoder.emb = nn.Embedding(VOCAB_SIZE, DECODER_EMBEDDING_DIM).to(DEVICE)
    decoder.vocab_projection = nn.Linear(DECODER_EMBEDDING_DIM, VOCAB_SIZE).to(DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Use tokenizer's pad token ID
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    # Special token IDs
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    special_token_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['additional_special_tokens'])

    # Training loop
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            mels, tokens, mel_lengths, token_lengths = batch
            mels = mels.to(DEVICE)  # Shape: (batch_size, n_mels, max_mel_length)
            tokens = tokens.to(DEVICE)  # Shape: (batch_size, max_token_length)

            # Shift tokens for decoder input and target
            decoder_input = tokens[:, :-1]  # Input to the decoder
            decoder_target = tokens[:, 1:]  # Target output

            # Forward pass through encoder
            encoder_outputs = encoder(mels)  # Shape: (batch_size, time_steps, encoder_embedding_dim)

            # Forward pass through decoder
            logits = decoder(decoder_input, encoder_outputs)  # Shape: (batch_size, seq_length, vocab_size)

            # Compute loss
            logits = logits.view(-1, VOCAB_SIZE)
            decoder_target = decoder_target.contiguous().view(-1)
            loss = criterion(logits, decoder_target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

        # Evaluation code
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            # Get a sample batch
            sample_batch = next(iter(dataloader))
            mels, tokens, mel_lengths, token_lengths = sample_batch
            mels = mels.to(DEVICE)
            tokens = tokens.to(DEVICE)

            # Prepare decoder input
            decoder_input = tokens[:, :-1]
            decoder_target = tokens[:, 1:]

            # Forward pass through encoder
            encoder_outputs = encoder(mels)

            # Inference: generate predictions
            # For simplicity, we'll use teacher forcing here
            logits = decoder(decoder_input, encoder_outputs)
            predictions = logits.argmax(dim=-1)

            for i in range(min(3, mels.size(0))):  # Display up to 3 examples
                pred_tokens = predictions[i].tolist()
                # Truncate at EOS token
                pred_tokens = truncate_at_eos(pred_tokens, eos_token_id)
                # Filter out special tokens
                pred_tokens_filtered = [t for t in pred_tokens if t not in special_token_ids + [bos_token_id, eos_token_id, tokenizer.pad_token_id]]
                # Decode prediction
                pred_caption = tokenizer.decode(pred_tokens_filtered).strip()

                # Target tokens
                target_tokens = decoder_target[i].tolist()
                target_tokens = truncate_at_eos(target_tokens, eos_token_id)
                target_tokens_filtered = [t for t in target_tokens if t not in special_token_ids + [bos_token_id, eos_token_id, tokenizer.pad_token_id]]
                target_caption = tokenizer.decode(target_tokens_filtered).strip()

                # Compute Precision, Recall, F1 Score
                same_tokens = sum(1 for pred, target in zip(pred_tokens_filtered, target_tokens_filtered) if pred == target)
                precision = same_tokens / len(pred_tokens_filtered) if pred_tokens_filtered else 0
                recall = same_tokens / len(target_tokens_filtered) if target_tokens_filtered else 0
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0

                # BLEU Score
                bleu_score = sentence_bleu([target_tokens_filtered], pred_tokens_filtered, weights=(0.5, 0.5))

                print(f"Sample {i + 1} - Predicted: {pred_caption}")
                print(f"Sample {i + 1} - Target:    {target_caption}")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, BLEU Score: {bleu_score:.4f}\n")

        encoder.train()
        decoder.train()

        # Optionally, save model checkpoints
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./models/checkpoint_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth")

if __name__ == "__main__":
    train()
