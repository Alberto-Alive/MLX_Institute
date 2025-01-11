# Good
#  import pandas as pd
# import numpy as np
# import json
# import random
# import time

# # Load Parquet data
# start_time = time.time()
# df = pd.read_parquet("./data/train_bing.parquet")
# print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# # List to hold cleaned triplets
# cleaned_data = []

# # Track processing time for rows
# start_processing_time = time.time()
# row_count = 0

# # Process each row to create triplets
# for _, row in df.iterrows():
#     query = row.query
#     positive_passages = row.passages["passage_text"]

#     # Skip if there are no positive passages in the current row
#     if not any(positive_passages):
#         continue

#     # Create triplets for each positive passage
#     for positive in positive_passages:
        
#         # Initialize a set to track (random_row_index, negative_passage_index) pairs
#         sampled_pairs = set()

#         for _ in range(100):
#             # Sample a random index for the negative passage row
#             random_index = random.randint(0, len(df) - 1)
#             random_row = df.iloc[random_index]
            
#             # Ensure that the random row's query is different from the current row's query
#             while random_row.query == query:
#                 random_index = random.randint(0, len(df) - 1)
#                 random_row = df.iloc[random_index]
            
#             # Get the negative passages from the sampled row
#             negative_passages = random_row.passages["passage_text"]
            
#             # Skip if there are no passages in the sampled row
#             if not any(negative_passages):
#                 continue
            
#             # Find indices of selectable negative passages
#             selectable_negatives = np.where(random_row.passages["is_selected"] == 1)[0]

#             # Try to pick a unique negative passage index within this row
#             negative_index = None
#             for idx in selectable_negatives:
#                 pair = (random_index, idx)
#                 if pair not in sampled_pairs:
#                     negative_index = idx
#                     sampled_pairs.add(pair)
#                     break

#             # If no unique negative passage was found in this row, skip to the next iteration
#             if negative_index is None:
#                 for idx in range(len(negative_passages)):
#                     pair = (random_index, idx)
#                     if pair not in sampled_pairs:
#                         negative_index = idx
#                         sampled_pairs.add(pair)
#                         break
#                 if negative_index is None:
#                     continue

#             # Select the negative passage text
#             negative = negative_passages[negative_index]

#             # Add the triplet to the cleaned data
#             cleaned_data.append([query, positive, negative])

#     # Print progress every 1000 rows
#     row_count += 1
#     if row_count % 1000 == 0:
#         print(f"Processed {row_count} rows")

# print(f"Total data processing time: {time.time() - start_processing_time:.2f} seconds")

# # Save the cleaned data
# start_save_time = time.time()
# with open("./data/cleaned_triplets_amplify.json", "w") as f:
#     json.dump(cleaned_data, f, indent=4)
# print(f"Data saving time: {time.time() - start_save_time:.2f} seconds")
# print(f"Total execution time: {time.time() - start_time:.2f} seconds")



# Better
# import pandas as pd
# import numpy as np
# import json
# import random
# import time

# # Load Parquet data
# start_time = time.time()
# df = pd.read_parquet("./data/train_bing.parquet")
# print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# # Flatten the nested passages into a DataFrame
# passages_list = []
# for idx, row in df.iterrows():
#     query = row.query
#     passage_texts = row.passages['passage_text']
#     is_selected = row.passages['is_selected']
#     for i in range(len(passage_texts)):
#         passages_list.append({
#             'query': query,
#             'passage_text': passage_texts[i],
#             'is_selected': is_selected[i]
#         })

# passages_df = pd.DataFrame(passages_list)
# passages_df.reset_index(drop=True, inplace=True)

# # Prepare for negative sampling
# total_passages = len(passages_df)

# # List to hold cleaned triplets
# cleaned_data = []

# # Track processing time for rows
# start_processing_time = time.time()
# row_count = 0

# # Process each row to create triplets
# for idx, row in df.iterrows():
#     query = row.query
#     passage_texts = row.passages['passage_text']
#     is_selected = row.passages['is_selected']
#     positive_indices = [i for i, selected in enumerate(is_selected) if selected == 1]

#     if not positive_indices:
#         continue

#     for pos_idx in positive_indices:
#         positive = passage_texts[pos_idx]

#         # Sample negative passages
#         # Adjust the sample size to ensure enough negatives after filtering
#         negative_candidates = passages_df.sample(n=30, replace=True)

#         # Filter out negatives with the same query
#         negative_candidates = negative_candidates[negative_candidates['query'] != query].head(10)

#         for _, negative_row in negative_candidates.iterrows():
#             negative = negative_row['passage_text']
#             cleaned_data.append([query, positive, negative])

#     # Print progress every 1000 rows
#     row_count += 1
#     if row_count % 1000 == 0:
#         print(f"Processed {row_count} rows")

# print(f"Total data processing time: {time.time() - start_processing_time:.2f} seconds")

# # Save the cleaned data
# start_save_time = time.time()
# with open("./data/cleaned_triplets_amplify.json", "w") as f:
#     json.dump(cleaned_data, f, indent=4)
# print(f"Data saving time: {time.time() - start_save_time:.2f} seconds")
# print(f"Total execution time: {time.time() - start_time:.2f} seconds")
