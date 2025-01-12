import sentencepiece as spm
import json

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('./models/wiki_tokenizer.model')


with open('./data/wiki_string.txt', 'r', encoding='utf-8') as file:
    wiki_text = file.read()


# Tokenize the text and convert to token IDs
token_ids = sp.encode(wiki_text, out_type=int)

chunk_size = 128  # Set to 256 if you want longer context

# Chunk the token IDs into segments of `chunk_size`
chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
# print("Chunks:", chunks)

# Create mappings for words to IDs and IDs to words
words_to_ids = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
ids_to_words = {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}

# Output the mappings
# print("words_to_ids:", words_to_ids)
# print("ids_to_words:", ids_to_words)


# Save chunks to a JSON file
with open('./data/chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

# Save words_to_ids and ids_to_words to JSON files
with open('./data/words_to_ids.json', 'w', encoding='utf-8') as f:
    json.dump(words_to_ids, f, ensure_ascii=False, indent=4)

with open('./data/ids_to_words.json', 'w', encoding='utf-8') as f:
    json.dump(ids_to_words, f, ensure_ascii=False, indent=4)

print("Chunks and mappings saved to files.")
    