import os
import librosa
import numpy as np
import sentencepiece as spm
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# Configuration
ROOT_DIR = r"./train-clean-100"  # Root directory for the dataset
SAVE_DIR = r"./preprocessed_dataset"  # Directory to save preprocessed data
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
SUBSET_SIZE = 100 


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

    # Normalize the log-Mel spectrogram to [-1, 1]
    log_mel_spec_min = log_mel_spec.min()
    log_mel_spec_max = log_mel_spec.max()
    log_mel_spec = 2 * ((log_mel_spec - log_mel_spec_min) / (log_mel_spec_max - log_mel_spec_min + 1e-6)) - 1

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
    tokens = tokenizer.encode(transcription)

    # Save data
    np.save(mel_save_path, log_mel_spec)
    torch.save(torch.tensor(tokens), tokens_save_path)


# 5. Train or Load Tokenizer
def train_or_load_tokenizer(transcriptions, model_prefix="tokenizer", vocab_size=5000):
    """
    Train a SentencePiece tokenizer if the model doesn't exist, otherwise load it.
    """
    model_file = f"{model_prefix}.model"

    # Check if the model already exists
    if os.path.exists(model_file):
        print(f"Loading existing tokenizer model from {model_file}")
        tokenizer = spm.SentencePieceProcessor(model_file=model_file)
        return tokenizer

    print("Training new tokenizer...")
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
    tokenizer = spm.SentencePieceProcessor(model_file=model_file)
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

    print(f"Found {len(pairs)} pairs.")

    # Step 2: Train or load tokenizer
    print("Training tokenizer...")
    all_transcriptions = {audio_path: text for audio_path, text in pairs}
    tokenizer = train_or_load_tokenizer(all_transcriptions)

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
