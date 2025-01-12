import sentencepiece as spm

# Load the SentencePiece model from the .model file
sp = spm.SentencePieceProcessor()
sp.load('./models/wiki_tokenizer.model')

# Tokenize text
tokens = sp.encode('This is a sample sentence.', out_type=str)
print("tokens", tokens)  # Outputs the tokenized sentence as a list of tokens
# Viewing the vocabulary file
with open('./models/wiki_tokenizer.vocab', 'r', encoding='utf-8') as f:
    vocab = f.readlines()

# Print the first 10 lines of the vocabulary
print("vocab", vocab[:10])  # Each line contains a token and its frequency
