import pandas as pd
import json
import random
import time

# Start timer for loading data
start_time = time.time()

# Load Parquet data
df = pd.read_parquet("train_bing.parquet")
load_time = time.time() - start_time
print(f"Data loading time: {load_time:.2f} seconds")

# List to hold cleaned triplets
cleaned_data = []

# Start timer for processing rows
start_processing_time = time.time()
row_count = 0  # Initialize row counter

# Loop through each row to process data
for _, row in df.iterrows():
    row_start_time = time.time()  # Timer for each row

    query = row.query
    positive_passages = row.passages["passage_text"]  # Treat all passages in this row as positive

    # Skip if there are no positive passages in the current row
    if any(positive_passages) == False:
        continue

    # Create triplets for each positive passage
    for positive in positive_passages:
        triplet_start_time = time.time()  # Timer for each triplet

        # Sample a random row as the source for the negative passage
        random_row = df.sample(n=1).iloc[0]
        
        # Ensure the negative passage comes from a different query
        while random_row.query == query:
            random_row = df.sample(n=1).iloc[0]  # Re-sample if it matches the current query
        
        # Select a random passage from the sampled row as the negative passage
        negative_passages = random_row.passages["passage_text"]
        
        # Skip if the random row has no passages
        if any(negative_passages) == False:
            continue
        
        # Randomly choose a negative passage from the sampled row
        negative = random.choice(negative_passages)
        
        # Add the triplet to the cleaned data
        cleaned_data.append([query, positive, negative])

        triplet_time = time.time() - triplet_start_time

    # Increment row counter
    row_count += 1

    # Print every 1000 rows
    if row_count % 1000 == 0:
        print(f"Processed {row_count} rows")

    row_time = time.time() - row_start_time

processing_time = time.time() - start_processing_time
print(f"Total data processing time: {processing_time:.2f} seconds")

# Start timer for saving data
start_save_time = time.time()

# Save the cleaned data to a JSON file
with open("cleaned_triplets.json", "w") as f:
    json.dump(cleaned_data, f, indent=4)

save_time = time.time() - start_save_time
print(f"Data saving time: {save_time:.2f} seconds")

# Total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")
