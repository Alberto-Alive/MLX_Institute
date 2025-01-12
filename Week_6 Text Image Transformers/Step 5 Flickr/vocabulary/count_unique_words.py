from collections import Counter
import re

# Path to your local captions.txt file
captions_file_path = '../data/captions_4_vocab.txt'

# Initialize a counter to count unique words
word_counter = Counter()

# Process each caption in the file
with open(captions_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Tokenize and clean each line
        words = re.findall(r'\b\w+\b', line.lower())  # Convert to lowercase and extract words
        word_counter.update(words)

# Count of unique words
unique_word_count = len(word_counter)
print(f"Unique words in captions.txt: {unique_word_count}")
