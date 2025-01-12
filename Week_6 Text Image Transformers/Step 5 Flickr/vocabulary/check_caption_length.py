import pandas as pd
import json

# Path to the CSV file containing the captions
annotations_file_path = '../data/flickr_annotations_30k.csv'

# Load the CSV file
df = pd.read_csv(annotations_file_path)

# Initialize a variable to store the maximum caption length
max_caption_length = 0

# Loop over each row to calculate the length of each caption and update the max length
for _, row in df.iterrows():
    captions = json.loads(row['raw'])  # Assuming 'raw' contains JSON-formatted list of captions
    for caption in captions:
        caption_length = len(caption.split())  # Calculate caption length by word count
        max_caption_length = max(max_caption_length, caption_length)

print(f"Maximum caption length (in words): {max_caption_length}")

caption_lengths = [
    len(caption.split()) 
    for _, row in df.iterrows() 
    for caption in json.loads(row['raw'])
]

# Calculate the 95th percentile of caption lengths
max_length_95th = int(pd.Series(caption_lengths).quantile(0.92))

print(f"95th percentile maximum caption length (in words): {max_length_95th}")