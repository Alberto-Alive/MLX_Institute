import pandas as pd

snli_df = pd.read_parquet('./data/snli.parquet')
print(snli_df.columns)
print(snli_df.head())

# Concatenate premise and hypothesis
snli_df['text'] = snli_df['premise'] + " " + snli_df['hypothesis']


import sentencepiece as spm
import random

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('./models/wiki_tokenizer.model')

# Define [MASK] token ID
mask_id = sp.piece_to_id("[MASK]")  # Ensure "[MASK]" is in the tokenizer vocabulary

# Function to tokenize and mask 15% of tokens
def tokenize_and_mask(text, mask_id, vocab_size):
    # Tokenize the text to get token IDs
    token_ids = sp.encode(text, out_type=int)
    
    # Mask 15% of the tokens as per BERT's MLM strategy
    num_to_mask = max(1, int(len(token_ids) * 0.15))
    mask_indices = random.sample(range(len(token_ids)), num_to_mask)

    labels = [-100] * len(token_ids)  # Initialize with -100 for ignored positions
    masked_token_ids = token_ids[:]

    for idx in mask_indices:
        if random.random() < 0.8:
            masked_token_ids[idx] = mask_id
        elif random.random() < 0.5:
            masked_token_ids[idx] = random.randint(0, vocab_size - 1)
        labels[idx] = token_ids[idx]

    return masked_token_ids, labels


# Apply the function to each row in the DataFrame
vocab_size = sp.get_piece_size()
snli_df[['masked_tokens', 'mlm_labels']] = snli_df['text'].apply(
    lambda x: pd.Series(tokenize_and_mask(x, mask_id, vocab_size))
)


# Save the processed DataFrame for training use
snli_df.to_parquet('./data/snli_mlm.parquet')

