import torch
import json
import sys
import random
import os
CURRENT_DIR = os.path.dirname(__file__) 

# Parameters
EMBEDDING_DIM = 64
STATE_DICT_PATH = os.path.join(CURRENT_DIR, 'FineTunedTwoTowerBase.pth')
VOCAB_PATH = os.path.join(CURRENT_DIR, 'fine_text8_words_to_ids.json')
MODEL_PATH = os.path.join(CURRENT_DIR, 'two_tower_model_gpu_bs500_epochs10_lr1e-05_loss0.55.pth')
PASSAGE_EMBS_PATH =  os.path.join(CURRENT_DIR, 'passage_embeddings.pth')
ALL_PASSAGES_PATH = os.path.join(CURRENT_DIR, 'all_passages.json')
UNK_TOKEN = '<UNK>'

# Function: Load the vocabulary and state dictionary
def load_state_dict_and_vocab():
    # Step 1: Load the state_dict and extract embedding weights
    state_dict = torch.load(STATE_DICT_PATH, map_location='cpu')
    if 'emb.weight' in state_dict:
        embedding_weights = state_dict['emb.weight']
    else:
        raise KeyError("The state_dict does not contain 'emb.weight'.")

    # Step 2: Load the vocabulary mapping
    with open(VOCAB_PATH, 'r') as f:
        word_to_idx = json.load(f)

    return embedding_weights, word_to_idx

# Function: Convert text to vector
def txt2vec(query, embedding_weights, word_to_idx):
    """
    Converts a text string into a mean embedding vector.
    
    Args:
        query (str): The input text.
    
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

# Define Two Tower Model Classes
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

# Function: Run the inference process
def run_two_tower_inference(query):
    # STAGE I: Load state_dict and vocabulary
    embedding_weights, word_to_idx = load_state_dict_and_vocab()

    # Convert query to vector using txt2vec
    query_vec = txt2vec(query, embedding_weights, word_to_idx)

    # Ensure the query vector has the correct dimension
    if query_vec.dim() == 1:
        query_vec = query_vec.unsqueeze(0)

    # STAGE II: Load and initialize the Two Tower model
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cpu")
    two_tower = TwoTower().to(device)

    # Load the state_dict for the Two Tower model
    state_dict = torch.load(MODEL_PATH, map_location=device)
    two_tower.load_state_dict(state_dict)
    two_tower.eval()

    # Get query embedding
    with torch.no_grad():
        query_embeddings = two_tower.qry_tower(query_vec)

    # Load passage embeddings
    passage_embeddings_dict = torch.load(PASSAGE_EMBS_PATH, map_location=device)
    passage_embeddings = passage_embeddings_dict['embeddings'].to(device)

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(query_embeddings, passage_embeddings)

    # Retrieve top 5 closest passage embeddings
    top_5_similarities, top_5_indices = torch.topk(similarity, 5)

    # Load all passages
    with open(ALL_PASSAGES_PATH, 'r') as file:
        passages = json.load(file)

    # Retrieve passages at the specified indexes
    indexes = top_5_indices.tolist()
    selected_passages = [passages[i] for i in indexes]

    return selected_passages, indexes

