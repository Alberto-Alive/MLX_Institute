import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import matplotlib.pyplot as plt
import torch

# Configuration
ROOT_DIR = r"./train-clean-100"  # Root directory for the dataset
SAVE_DIR = r"./processed_dataset"  # Directory to save preprocessed data
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 8

# 1. Preprocessing Function for Audio
def preprocess_audio(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert an audio file to a log-Mel spectrogram.
    """
    # Load the audio file and resample
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Convert to log scale
    log_mel_spec = np.log1p(mel_spec)
    return log_mel_spec

# 2. Parse Transcriptions
def load_transcriptions(transcription_file):
    """
    Load the transcription file and map audio IDs to transcriptions.
    """
    transcriptions = {}
    with open(transcription_file, "r") as f:
        for line in f:
            # Split each line into the file ID and the transcription
            file_id, transcription = line.strip().split(" ", 1)
            transcriptions[file_id] = transcription
    return transcriptions

# 3. Find Audio Files and Pair with Transcriptions
def find_audio_transcription_pairs(root_dir):
    """
    Find all .flac audio files and pair them with their transcriptions.
    """
    pairs = []
    for root, _, files in os.walk(root_dir):
        # Check if a transcription file exists in the current folder
        trans_files = [f for f in files if f.endswith(".trans.txt")]
        for trans_file in trans_files:
            trans_file_path = os.path.join(root, trans_file)
            transcriptions = load_transcriptions(trans_file_path)

            # Find .flac files in the same folder
            for file in files:
                if file.endswith(".flac"):
                    audio_path = os.path.join(root, file)
                    file_id = os.path.splitext(file)[0]  # Extract file ID (e.g., "8975-270782-0000")

                    # Check if the file ID has a corresponding transcription
                    if file_id not in transcriptions:
                        print(f"Warning: No transcription found for {audio_path}")
                        continue

                    # Pair the audio file with its transcription
                    pairs.append((audio_path, transcriptions[file_id]))
    return pairs

# 4. Dataset Class
class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech data.
    """
    def __init__(self, audio_transcription_pairs, tokenizer):
        self.pairs = audio_transcription_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get the audio file and transcription
        audio_file, transcription = self.pairs[idx]

        # Preprocess audio
        log_mel_spec = preprocess_audio(audio_file)

        # Tokenize transcription
        tokens = self.tokenizer.encode(transcription)

        return torch.tensor(log_mel_spec, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

# 5. Tokenizer
def train_or_load_tokenizer(transcriptions, model_prefix="tokenizer", vocab_size=5000):
    """
    Train or load a SentencePiece tokenizer.
    """
    # Save transcriptions to a temporary file for tokenizer training
    with open("temp_transcriptions.txt", "w") as f:
        for _, text in transcriptions.items():
            f.write(text.strip() + "\n")

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input="temp_transcriptions.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size
    )

    # Load the trained tokenizer
    tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    return tokenizer

def save_dataset(dataset, save_dir=SAVE_DIR):
    """
    Save the preprocessed dataset (audio features and tokens) to disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (log_mel_spec, tokens) in enumerate(dataset):
        # Save log-Mel spectrogram as a .npy file
        mel_save_path = os.path.join(save_dir, f"mel_{idx}.npy")
        np.save(mel_save_path, log_mel_spec)

        # Save tokens as a .pt file
        tokens_save_path = os.path.join(save_dir, f"tokens_{idx}.pt")
        torch.save(tokens, tokens_save_path)

    print(f"Saved {len(dataset)} examples to {save_dir}")

# 6. Main Function
def main():
    # Step 1: Find audio-transcription pairs
    print("Finding audio-transcription pairs...")
    pairs = find_audio_transcription_pairs(ROOT_DIR)
    print(f"Found {len(pairs)} pairs.")

    # Step 2: Train or load tokenizer
    print("Training tokenizer...")
    all_transcriptions = {audio_path: text for audio_path, text in pairs}
    tokenizer = train_or_load_tokenizer(all_transcriptions)

    # Step 3: Create dataset
    print("Creating dataset...")
    dataset = LibriSpeechDataset(pairs, tokenizer)

    save_dataset(dataset)

    # Step 4: Visualize an example
    log_mel_spec, tokens = dataset[0]
    print("First Example:")
    print("Log-Mel Spectrogram Shape:", log_mel_spec.shape)
    print("Tokens:", tokens)

    # Optional: Visualize the log-Mel spectrogram
    plt.imshow(log_mel_spec.numpy(), aspect="auto", origin="lower", cmap="viridis")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.colorbar(label="Log Energy")
    plt.show()

    # Step 5: Use DataLoader for batching
    print("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Iterate through a batch
    for batch_idx, (mel_specs, token_batches) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Mel Spectrograms Shape:", mel_specs.shape)
        print("Token Batches Shape:", token_batches.shape)
        break

# Run the script
if __name__ == "__main__":
    main()
