# import sentencepiece as spm

# # Train SentencePiece model
# spm.SentencePieceTrainer.train(
#     input='../data/captions_4_vocab.txt',        # Path to the caption text file
#     model_prefix='../models/caption_tokenizer',  # Prefix for the model files
#     vocab_size=16458              # Adjust vocabulary size based on your needs
# )


## Add special tokens lol

import sentencepiece as spm

# Define only the custom special token
special_tokens = ['<pad>']

# Train SentencePiece model with the custom special token
spm.SentencePieceTrainer.train(
    input='../data/captions_4_vocab.txt',                # Path to the caption text file
    model_prefix='../models/caption_tokenizer_updated',  # Prefix for the model files
    vocab_size=16458,                                    # Adjust vocabulary size based on your needs
    pad_id=0,                                            # Assign ID 0 to <pad>
    unk_id=1,                                            # Assign ID 1 to <unk>
    bos_id=2,                                            # Assign ID 2 to <s>
    eos_id=3,                                            # Assign ID 3 to </s>
)


tokenizer = spm.SentencePieceProcessor()
tokenizer.load('../models/caption_tokenizer_updated.model')

# Define the special tokens
special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
print("PAD ID:", tokenizer.pad_id())  # Should print 0
print("VOCAB SIZE:", tokenizer.get_piece_size())  # Ensure it matches your expected size


print("Special Tokens and Their IDs:")
for token in special_tokens:
    token_id = tokenizer.piece_to_id(token)
    print(f" - Token '{token}': ID {token_id}")