import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# Configuration
ROOT_DIR = r"./train-clean-100"  # Root directory for the dataset
SAVE_DIR = r"./multi_processed_dataset"  # Directory to save preprocessed data
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
BATCH_SIZE = 8
SUBSET_SIZE = 50  # Process a smaller subset for testing
MAX_LEN = 1000  # Maximum allowed spectrogram length


# 1. Preprocessing Function for Audio
def preprocess_audio(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Convert an audio file to a log-Mel spectrogram.
    """
    # Load the audio file and resample
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

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
            if not line.strip():  # Skip empty lines
                continue
            try:
                file_id, transcription = line.strip().split(" ", 1)
                transcriptions[file_id] = transcription
            except ValueError:
                print(f"Warning: Malformed line in {transcription_file}: {line}")
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


# 4. Dataset Class with Multiprocessing
class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech data with parallel preprocessing.
    """
    def __init__(self, audio_transcription_pairs, tokenizer, subset_size=None, save_dir=SAVE_DIR):
        self.pairs = audio_transcription_pairs[:subset_size] if subset_size else audio_transcription_pairs
        self.tokenizer = tokenizer
        self.save_dir = save_dir

        # Preprocess data in parallel
        print("Preprocessing dataset in parallel...")
        self.preprocess_and_save()

    def preprocess_and_save(self):
        """
        Preprocess audio files and save them in parallel.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        Parallel(n_jobs=-1, backend="threading")(
            delayed(self.process_single_file)(idx, audio_path, transcription)
            for idx, (audio_path, transcription) in tqdm(enumerate(self.pairs), total=len(self.pairs))
        )

    def process_single_file(self, idx, audio_path, transcription):
        """
        Process a single audio-transcription pair and save results.
        """
        # File paths
        mel_save_path = os.path.join(self.save_dir, f"mel_{idx}.npy")
        tokens_save_path = os.path.join(self.save_dir, f"tokens_{idx}.pt")

        # Skip if already processed
        if os.path.exists(mel_save_path) and os.path.exists(tokens_save_path):
            return

        # Preprocess audio and tokenize transcription
        log_mel_spec = preprocess_audio(audio_path)
        if log_mel_spec.shape[1] < 10:  # Skip very short files
            print(f"Warning: Skipping short audio file {audio_path}")
            return
        tokens = self.tokenizer.encode(transcription)

        # Save results
        np.save(mel_save_path, log_mel_spec)
        torch.save(torch.tensor(tokens), tokens_save_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Load preprocessed data
        mel_save_path = os.path.join(self.save_dir, f"mel_{idx}.npy")
        tokens_save_path = os.path.join(self.save_dir, f"tokens_{idx}.pt")

        log_mel_spec = np.load(mel_save_path)
        tokens = torch.load(tokens_save_path)

        # Pad the spectrogram along the time dimension
        padded_spec = F.pad(
            torch.tensor(log_mel_spec, dtype=torch.float32),
            (0, MAX_LEN - log_mel_spec.shape[1]),  # Pad on the right
            value=0.0  # Use zero-padding
        )

        return padded_spec[:, :MAX_LEN], tokens


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


def collate_fn(batch):
    """
    Dynamically pad spectrograms and tokens to the longest length in the batch.
    """
    specs, tokens = zip(*batch)

    # Pad spectrograms
    max_spec_len = max(spec.shape[1] for spec in specs)
    padded_specs = torch.stack([
        F.pad(spec, (0, max_spec_len - spec.shape[1]), value=0.0) for spec in specs
    ])

    # Pad tokens
    max_token_len = max(len(token) for token in tokens)
    padded_tokens = torch.stack([
        F.pad(token, (0, max_token_len - len(token)), value=0) for token in tokens
    ])

    return padded_specs, padded_tokens


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

    # Step 3: Create dataset with multiprocessing
    print("Creating dataset...")
    dataset = LibriSpeechDataset(pairs, tokenizer, subset_size=SUBSET_SIZE)

    # Step 4: Use DataLoader for batching
    print("Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Iterate through a batch
    for batch_idx, (mel_specs, token_batches) in enumerate(tqdm(dataloader, desc="Processing Batches")):
        print(f"Batch {batch_idx + 1}:")
        print("Mel Spectrograms Shape:", mel_specs.shape)
        print("Token Batches Shape:", token_batches.shape)
        break


# Run the script
if __name__ == "__main__":
    main()
