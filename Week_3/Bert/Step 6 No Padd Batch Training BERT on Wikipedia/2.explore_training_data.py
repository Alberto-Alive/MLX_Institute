import json

with open("./data/training_data.json", "r") as f:
    loaded_training_data = json.load(f)

with open("./data/vocabulary.json", "r") as f:
    loaded_vocab = json.load(f)


# Check the type of the dataset (usually a list or dictionary)
print(type(loaded_training_data))

# Print the number of samples
print(f"Total samples: {len(loaded_training_data)}")

# Print the first few samples to inspect the structure
# for i in range(3):  # Adjust range to view more samples if needed
#     print(f"Sample {i + 1}: {loaded_training_data[i]}")

vocab = {word: idx for word, idx in loaded_vocab.items()}
vocab_size = len(vocab)
idx_to_vocab = {idx: word for word, idx in vocab.items()}

for i, (masked_sequence, target_tokens) in enumerate(loaded_training_data):
    if len(masked_sequence) != 128:
        print(f"masked_sequence in sequence {i} length: {len(masked_sequence)}")
    elif len(target_tokens) != 20:
        print(f"target_tokens in sequence {i} length: {len(target_tokens)}")

        
    # print(f"target_tokens in sequence {i} length: {len(target_tokens)}")
    # print([idx_to_vocab[word_idx] for word_idx in masked_sequence])
    # print("len of masked_sequence: ",len(masked_sequence))
    # print("len of target_tokens: ",len(target_tokens))
    # break

