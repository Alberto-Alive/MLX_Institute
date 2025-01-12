import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm

# Load Parquet data
start_time = time.time()
df = pd.read_parquet("./data/train_bing.parquet")
print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# Task 1: Index Preparation
# Flatten the nested passages into a DataFrame and save necessary index data
passages_list = []
index_data = []

# Use tqdm to add a progress bar for iterating over rows
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Index Preparation"):
    query = row.query
    passage_texts = row.passages['passage_text']
    is_selected = row.passages['is_selected']

    # Save passage details for negative sampling
    for i in range(len(passage_texts)):
        passages_list.append({
            'query': query,
            'passage_text': passage_texts[i],
            'is_selected': is_selected[i]
        })

    # Save the positive passage indexes and query for creating triplets
    positive_indices = [i for i, selected in enumerate(is_selected) if selected == 1]
    if positive_indices:
        # Convert ndarray to list for JSON serialization
        index_data.append({
            'query': query,
            'positive_indices': positive_indices,
            'passage_texts': passage_texts.tolist()  # Convert ndarray to list
        })

# Convert passages to a DataFrame
passages_df = pd.DataFrame(passages_list)
passages_df.reset_index(drop=True, inplace=True)

# Save index data for triplet creation
with open("./data/index_data.json", "w") as f:
    json.dump(index_data, f, indent=4)

print(f"Indexing preparation time: {time.time() - start_time:.2f} seconds")

# Task 2: Triplet Creation
start_processing_time = time.time()
cleaned_data = []

# Load index data
with open("./data/index_data.json", "r") as f:
    index_data = json.load(f)

# Use tqdm to add a progress bar for creating triplets
for row in tqdm(index_data, desc="Triplet Creation"):
    query = row['query']
    positive_indices = row['positive_indices']
    passage_texts = row['passage_texts']

    for pos_idx in positive_indices:
        positive = passage_texts[pos_idx]

        # Sample negative passages
        negative_candidates = passages_df.sample(n=600, replace=True)

        # Filter out negatives with the same query
        negative_candidates = negative_candidates[negative_candidates['query'] != query].head(300)

        for _, negative_row in negative_candidates.iterrows():
            negative = negative_row['passage_text']
            cleaned_data.append([query, positive, negative])

# Save the cleaned data
start_save_time = time.time()
with open("./data/cleaned_triplets_stages.json", "w") as f:
    json.dump(cleaned_data, f, indent=4)
print(f"Data saving time: {time.time() - start_save_time:.2f} seconds")
print(f"Total triplet creation time: {time.time() - start_processing_time:.2f} seconds")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
