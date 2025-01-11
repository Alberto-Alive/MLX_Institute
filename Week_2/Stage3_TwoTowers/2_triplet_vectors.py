import torch
import json
from tqdm import tqdm  # For progress bar

# Parameters
EMBEDDING_DIM = 64
STATE_DICT_PATH = 'FineTunedTwoTowerBase.pth'
VOCAB_PATH = './fine_text8_words_to_ids.json'
TRIPLET_PATH = 'cleaned_triplets.json'
SAVE_PATH = 'triplet_vectors.pth'
UNK_TOKEN = '<UNK>'

# Step 1: Load the state_dict and extract embedding weights
state_dict = torch.load(STATE_DICT_PATH, map_location='cpu')
print(f"Loaded state_dict keys: {state_dict.keys()}")  # Should show ['emb.weight', 'ffw.weight']

if 'emb.weight' in state_dict:
    embedding_weights = state_dict['emb.weight']  # Shape: [VOCAB_SIZE, EMBEDDING_DIM]
    print(f"Embedding weights shape: {embedding_weights.shape}")
else:
    raise KeyError("The state_dict does not contain 'emb.weight'.")

# Step 2: Load the vocabulary mapping
with open(VOCAB_PATH, 'r') as f:
    word_to_idx = json.load(f)
print(f"Vocabulary size loaded: {len(word_to_idx)}")

# Step 3: Define the txt2vec function
def txt2vec(txt):
    """
    Converts a text string into a mean embedding vector.
    
    Args:
        txt (str): The input text.
    
    Returns:
        torch.Tensor: A tensor of shape [EMBEDDING_DIM].
    """
    words = txt.lower().split()
    # Map words to indices, using <UNK> for OOV words
    indices = [word_to_idx.get(word, word_to_idx[UNK_TOKEN]) for word in words]
    
    if indices:
        vectors = embedding_weights[indices]  # Shape: [num_words, EMBEDDING_DIM]
        mean_vector = vectors.mean(dim=0)    # Shape: [EMBEDDING_DIM]
        return mean_vector
    else:
        return torch.zeros(EMBEDDING_DIM)

# Step 4: Load and process triplets
with open(TRIPLET_PATH, 'r') as f:
    cleaned_triplets = json.load(f)
print(f"Total triplets loaded: {len(cleaned_triplets)}")

qry_vectors = []
pos_vectors = []
neg_vectors = []

print("Converting triplets to vectors...")
for triplet in tqdm(cleaned_triplets, desc='Processing triplets'):
    qry_txt, pos_txt, neg_txt = triplet
    qry_vec = txt2vec(qry_txt)
    pos_vec = txt2vec(pos_txt)
    neg_vec = txt2vec(neg_txt)
    
    qry_vectors.append(qry_vec)
    pos_vectors.append(pos_vec)
    neg_vectors.append(neg_vec)

# Stack vectors into tensors
qry_tensor = torch.stack(qry_vectors)    # Shape: [N, EMBEDDING_DIM]
pos_tensor = torch.stack(pos_vectors)    # Shape: [N, EMBEDDING_DIM]
neg_tensor = torch.stack(neg_vectors)    # Shape: [N, EMBEDDING_DIM]

print(f"Queries tensor shape: {qry_tensor.shape}")
print(f"Positives tensor shape: {pos_tensor.shape}")
print(f"Negatives tensor shape: {neg_tensor.shape}")

# Step 5: Save the vectors to a file
torch.save({
    'queries': qry_tensor,
    'positives': pos_tensor,
    'negatives': neg_tensor
}, SAVE_PATH)

print(f"All triplet vectors saved to {SAVE_PATH}")
