# cleaner.py
# preprocessing.py

import os
import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Configuration
ROOT_DIR = r"./train-clean-100"  # Root directory for the dataset
SAVE_DIR = r"./preprocessed_dataset"  # Directory to save preprocessed data
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
SUBSET_SIZE = 100 

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

    # Normalize the log-Mel spectrogram to [-1, 1]
    log_mel_spec_min = log_mel_spec.min()
    log_mel_spec_max = log_mel_spec.max()
    log_mel_spec = 2 * ((log_mel_spec - log_mel_spec_min) / (log_mel_spec_max - log_mel_spec_min + 1e-6)) - 1

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

# 4. Save Preprocessed Data
def save_preprocessed_data(idx, audio_path, transcription, tokenizer):
    """
    Preprocess and save a single audio-transcription pair.
    """
    # File paths for saving
    mel_save_path = os.path.join(SAVE_DIR, f"mel_{idx}.npy")
    tokens_save_path = os.path.join(SAVE_DIR, f"tokens_{idx}.pt")

    # Skip if already processed
    if os.path.exists(mel_save_path) and os.path.exists(tokens_save_path):
        return

    # Preprocess audio
    log_mel_spec = preprocess_audio(audio_path)
    if log_mel_spec.shape[1] < 10:  # Skip very short files
        print(f"Warning: Skipping short audio file {audio_path}")
        return

    # Tokenize transcription
    # Prepare decoder input sequence with special tokens
    # For example, for transcription without timestamps:
    decoder_input = (
        [tokenizer.bos_token_id] +
        tokenizer.convert_tokens_to_ids(['<|transcribe|>', '<|notimestamps|>']) +
        tokenizer.encode(transcription, add_special_tokens=False) +
        [tokenizer.eos_token_id]
    )

    # Save data
    np.save(mel_save_path, log_mel_spec)
    torch.save(torch.tensor(decoder_input), tokens_save_path)

# 5. Load Tokenizer
def load_tokenizer():
    """
    Load the GPT2 tokenizer and add special tokens as needed.
    """
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Add special tokens
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.save_pretrained('tokenizer/') 

    return tokenizer

# 6. Main Function
def main():
    # Step 1: Find audio-transcription pairs
    print("Finding audio-transcription pairs...")
    pairs = find_audio_transcription_pairs(ROOT_DIR)
    print(f"Found {len(pairs)} pairs.")

    if SUBSET_SIZE:
        pairs = pairs[:SUBSET_SIZE]
        print(f"Using a subset of size {len(pairs)} pairs.")

    # Step 2: Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Step 3: Create output directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Step 4: Preprocess and save all pairs in parallel
    print("Preprocessing and saving data...")
    Parallel(n_jobs=-1, backend="threading")(
        delayed(save_preprocessed_data)(idx, audio_path, transcription, tokenizer)
        for idx, (audio_path, transcription) in tqdm(enumerate(pairs), total=len(pairs))
    )

    print(f"Preprocessed data saved to {SAVE_DIR}.")

# Run the script
if __name__ == "__main__":
    main()
