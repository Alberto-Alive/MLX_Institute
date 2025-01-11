import json
import random
import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('./models/wiki_tokenizer.model')

# Load the tokenized chunks from JSON file
with open('./data/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Define the special token ID for [MASK] (ensure [MASK] is in your vocabulary)
mask_id = sp.piece_to_id("[MASK]")  # Add "[MASK]" as a user-defined symbol in SentencePiece if it's not present

# Function to mask 15% of tokens in a chunk
def mask_tokens(chunk, mask_id, vocab_size):
    masked_chunk = chunk.copy()
    labels = [-100] * len(chunk)  # Initialize labels for MLM (-100 means ignore in loss)

    # Determine the number of tokens to mask
    num_to_mask = max(1, int(len(chunk) * 0.15))
    mask_indices = random.sample(range(len(chunk)), num_to_mask)

    for idx in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_chunk[idx] = mask_id
        # 10% of the time, replace with a random token
        elif random.random() < 0.5:
            masked_chunk[idx] = random.randint(0, vocab_size - 1)
        # 10% of the time, keep the original token

        # Set the label to the original token ID for MLM training
        labels[idx] = chunk[idx]

    return masked_chunk, labels

# Apply masking to each chunk
masked_chunks = []
labels_chunks = []
for chunk in chunks:
    masked_chunk, labels = mask_tokens(chunk, mask_id, sp.get_piece_size())
    masked_chunks.append(masked_chunk)
    labels_chunks.append(labels)

# Optional: Save the masked chunks and labels to files for later use
with open('./data/masked_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(masked_chunks, f, ensure_ascii=False, indent=4)

with open('./data/labels_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(labels_chunks, f, ensure_ascii=False, indent=4)

print("Masked chunks and labels saved to './data/masked_chunks.json' and './data/labels_chunks.json'.")
