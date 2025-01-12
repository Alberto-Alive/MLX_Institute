import sentencepiece as spm

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("../models/caption_tokenizer.model")

# Tokenize a sample caption
sample_caption = "Two young guys with shaggy hair look at their hands."
tokens = sp.encode(sample_caption, out_type=str)  # out_type=str to get string tokens
print("Tokenized caption:", tokens)


# Define the padding token
pad_token = '<pad>'

# Get the ID for the padding token
pad_id = sp.piece_to_id(pad_token)

# Check if the pad token exists
if pad_id == sp.unk_id():
    print(f"Padding token '{pad_token}' is not found in the tokenizer vocabulary.")
else:
    print(f"Padding token '{pad_token}' has ID: {pad_id}")


# Define the special tokens
special_tokens = ['<pad>', '<s>', '</s>', '<unk>']  # Include <unk> for unknown tokens

print("Special Tokens and Their IDs:")
for token in special_tokens:
    token_id = sp.piece_to_id(token)
    if token_id == sp.unk_id() and token != '<unk>':
        print(f" - Token '{token}': Not found in the vocabulary.")
    else:
        print(f" - Token '{token}': ID {token_id}")