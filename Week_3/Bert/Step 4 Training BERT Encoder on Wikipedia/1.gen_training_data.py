import os
import pandas as pd
import sentencepiece as spm
import random
import json
import torch
from tqdm import tqdm

# Step 1: Load the dataset
data = pd.read_parquet('./data/wikitext-2-v1.parquet')

# Combine all text entries into a single string
long_text = " ".join(data['text'].tolist())

# Step 2: Save text to a temporary file for SentencePiece training
with open("./data/wikipedia_corpus.txt", "w", encoding='utf-8') as f:
    f.write(long_text)

# Step 3: Train SentencePiece tokenizer with optional parameters
spm.SentencePieceTrainer.train(
    input='./data/wikipedia_corpus.txt',
    model_prefix='./models/wiki_tokenizer',
    vocab_size=16000,
    character_coverage=0.9995,
    user_defined_symbols=['[MASK]'],
    unk_piece='[UNK]'
)

# Load the tokenizer
sp = spm.SentencePieceProcessor(model_file='./models/wiki_tokenizer.model')

# Define MASK_TOKEN
MASK_TOKEN = sp.PieceToId('[MASK]')

# Step 4: Function to generate masked samples with a longer sequence length
def generate_masked_samples(text, tokenizer, max_seq_length=128, mask_prob=0.15):
    tokens = tokenizer.EncodeAsIds(text)  # Tokenize entire text
    samples = []
    
    # Split tokens into chunks of max_seq_length
    for i in range(0, len(tokens), max_seq_length):
        sequence = tokens[i:i + max_seq_length]
        
        # Pad the sequence if it's shorter than max_seq_length
        if len(sequence) < max_seq_length:
            sequence += [0] * (max_seq_length - len(sequence))

        # Apply random masking
        masked_sequence = sequence[:]
        target_tokens = []
        for j in range(len(masked_sequence)):
            if random.random() < mask_prob:
                target_tokens.append((j, masked_sequence[j]))  # Save original token
                masked_sequence[j] = MASK_TOKEN  # Mask the token

        # Only save samples where at least one token was masked
        if target_tokens:
            samples.append((masked_sequence, target_tokens))
        
    return samples

# Step 5: Generate training data for the entire long text
print("Generating masked samples from the text...")
training_data = generate_masked_samples(long_text, sp, max_seq_length=128)

# Step 6: Save training data and vocabulary to files with better formatting

# Save training data
with open("./data/training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

# Save vocabulary to file
vocab = {sp.IdToPiece(i): i for i in range(sp.vocab_size())}
with open("./data/vocabulary.json", "w") as f:
    json.dump(vocab, f, indent=2)

print("Training data and vocabulary saved.")

# Example of loading the training data and vocabulary in a future script
# Load training data
with open("./data/training_data.json", "r") as f:
    loaded_training_data = json.load(f)

# Load vocabulary
with open("./data/vocabulary.json", "r") as f:
    loaded_vocab = json.load(f)

# Convert vocabulary to PyTorch format if needed (optional)
torch_vocab = {word: torch.tensor(id) for word, id in loaded_vocab.items()}

print("Loaded training data and vocabulary.")
