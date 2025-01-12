import torch
import json
from tqdm import tqdm  # For progress bar

# Parameters
EMBEDDING_DIM = 64
STATE_DICT_PATH = './models/FineTunedTwoTowerBase.pth'
VOCAB_PATH = './data/fine_text8_words_to_ids.json'
PASSAGES_PATH = './data/all_passages.json'
SAVE_PATH = './data/all_passages_vectors.pth'
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

# Step 4: Load and process passages
with open(PASSAGES_PATH, 'r') as f:
    all_passages = json.load(f)
print(f"Total passages loaded: {len(all_passages)}")

passage_vectors = []

print("Converting passages to vectors...")
for passage in tqdm(all_passages, desc='Processing passages'):
    passage_vec = txt2vec(passage)
    passage_vectors.append(passage_vec)

# Stack vectors into a tensor
passage_tensor = torch.stack(passage_vectors)    # Shape: [N, EMBEDDING_DIM]

print(f"Passages tensor shape: {passage_tensor.shape}")

# Step 5: Save the vectors to a file
torch.save({
    'passages': passage_tensor
}, SAVE_PATH)

print(f"All passage vectors saved to {SAVE_PATH}")
