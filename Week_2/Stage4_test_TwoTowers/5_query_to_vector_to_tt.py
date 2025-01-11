import torch
import json
import sys
import random

# Parameters
EMBEDDING_DIM = 64
STATE_DICT_PATH = './models/FineTunedTwoTowerBase.pth'
VOCAB_PATH = './data/fine_text8_words_to_ids.json'
PASSAGES_PATH = './data/all_passages.json'
SAVE_PATH = './data/all_passages_vectors.pth'
UNK_TOKEN = '<UNK>'

# STAGE I

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
def txt2vec(query):
    """
    Converts a text string into a mean embedding vector.
    
    Args:
        txt (str): The input text.
    
    Returns:
        torch.Tensor: A tensor of shape [EMBEDDING_DIM].
    """
    words = query.lower().split()
    # Map words to indices, using <UNK> for OOV words
    indices = [word_to_idx.get(word, word_to_idx[UNK_TOKEN]) for word in words]
    
    if indices:
        vectors = embedding_weights[indices]  # Shape: [num_words, EMBEDDING_DIM]
        mean_vector = vectors.mean(dim=0)    # Shape: [EMBEDDING_DIM]
        return mean_vector
    else:
        return torch.zeros(EMBEDDING_DIM)


query = "concrete pads cost"  # Define a sample query string
query_vec = txt2vec(query)


# STAGE II

torch.manual_seed(42)
random.seed(42)

# Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f'----------Using device: {device}')

# Define your models
class QryTower(torch.nn.Module):
    def __init__(self):
        super(QryTower, self).__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        y = torch.nn.functional.relu(y)
        y = self.fc3(y)
        y = torch.nn.functional.relu(y)
        y = self.fc4(y)
        return y

class DocTower(torch.nn.Module):
    def __init__(self):
        super(DocTower, self).__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 16)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        y = torch.nn.functional.relu(y)
        y = self.fc3(y)
        y = torch.nn.functional.relu(y)
        y = self.fc4(y)
        return y

class TwoTower(torch.nn.Module):
    def __init__(self):
        super(TwoTower, self).__init__()
        self.qry_tower = QryTower()
        self.doc_tower = DocTower()

    def forward(self, qry, doc):
        qry_emb = self.qry_tower(qry)
        doc_emb = self.doc_tower(doc)
        return qry_emb, doc_emb




# Initialize your model (ensure it matches your saved model's architecture)
two_tower = TwoTower().to(device)
model_path = './models/two_tower_model_gpu_bs500_epochs10_lr1e-05_loss0.55.pth'

# Load the state_dict
state_dict = torch.load(model_path, map_location=device)
two_tower.load_state_dict(state_dict)
print("Model state_dict loaded successfully.")

# Set model to evaluation mode
two_tower.eval()

if query_vec.dim() == 1:
    query_vec = query_vec.unsqueeze(0)

with torch.no_grad():
    query_embeddings = two_tower.qry_tower(query_vec)

print("Sample embedding:", query_embeddings)


# Load the passage embeddings
passage_embeddings_dict = torch.load("./data/passage_embeddings.pth", map_location=device)
passage_embeddings = passage_embeddings_dict['embeddings'].to(device)

# Compute cosine similarity between query and passage embeddings
cos = torch.nn.CosineSimilarity(dim=0)
similarity = cos(query_embeddings, passage_embeddings)
# Retrieve 5 closest to my query embeddings
top_5_similarities, top_5_indices = torch.topk(similarity, 5)
print(f"Top 5 closest passage embeddings to the query: indices={top_5_indices}, similarities={top_5_similarities}")


with open('./data/all_passages.json', 'r') as file:
    passages = json.load(file)


# List of indexes you want to retrieve
indexes = top_5_indices.tolist() # replace with your specific indexes

# Retrieve passages at the specified indexes
selected_passages = [passages[i] for i in indexes]

print("Selected Passages:")
for idx, passage in zip(indexes, selected_passages):
    print(f"Index {idx}: {passage}")