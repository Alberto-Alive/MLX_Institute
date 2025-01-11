import torch
import json
import random
import time
import os
from .ml_model import txt2vec

CURRENT_DIR = os.path.dirname(__file__) 

torch.manual_seed(42)
random.seed(42)


EMBEDDING_DIM = 64
STATE_DICT_PATH = os.path.join(CURRENT_DIR, 'FineTunedTwoTowerBase.pth')
VOCAB_PATH = os.path.join(CURRENT_DIR, 'fine_text8_words_to_ids.json')
MODEL_PATH = os.path.join(CURRENT_DIR, 'two_tower_model_gpu_bs500_epochs10_lr1e-05_loss0.55.pth')
PASSAGE_EMBS_PATH =  os.path.join(CURRENT_DIR, 'passage_embeddings.pth')
ALL_PASSAGES_PATH = os.path.join(CURRENT_DIR, 'all_passages.json')
ALL_PASSAGES_VECTORS_PATH = os.path.join(CURRENT_DIR, 'all_passages_vectors.pth')
UPDATED_MODEL_PATH = os.path.join(CURRENT_DIR, 'updated_two_tower_model.pth')


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
  

def update_model(query, pos_pass_idxs, neg_pass_idxs):
    # Initialize model, criterion, and optimizer
    two_tower = TwoTower()
    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(two_tower.parameters(), lr=0.0001)
    print(pos_pass_idxs, neg_pass_idxs)
    state_dict = torch.load(STATE_DICT_PATH, map_location='cpu')
    if 'emb.weight' in state_dict:
        embedding_weights = state_dict['emb.weight']
    else:
        raise KeyError("The state_dict does not contain 'emb.weight'.")

    # Step 2: Load the vocabulary mapping
    with open(VOCAB_PATH, 'r') as f:
        word_to_idx = json.load(f)
    # qry = txt2vec(query, embedding_weights, word_to_idx).mean(dim=0)
    qry = txt2vec(query, embedding_weights, word_to_idx)
    print(f"Shape of qry after txt2vec and mean: {qry.shape}")

    qry = qry.unsqueeze(0)
    print(f"Shape of qry after unsqueeze: {qry.shape}")

    # Load the saved triplet vectors
    all_passages_vectors_dict = torch.load(ALL_PASSAGES_VECTORS_PATH)
    all_passages_vectors = all_passages_vectors_dict['passages']
    

    for i, pos_pass_idx in enumerate(pos_pass_idxs):
        pos_tensor = all_passages_vectors[pos_pass_idx]    # Shape: [N, 64]
        for j, neg_pass_idx in enumerate(neg_pass_idxs):
            neg_tensor = all_passages_vectors[neg_pass_idx]    # Shape: [N, 64]
            # qry = qry_tensor[i].unsqueeze(0)  # Shape: [1, 64]
            pos = pos_tensor.unsqueeze(0)  # Shape: [1, 64]
            neg = neg_tensor.unsqueeze(0)  # Shape: [1, 64]
            optimizer.zero_grad()
            qry_emb, pos_emb = two_tower(qry, pos)
            _, neg_emb = two_tower(qry, neg)
            pos_loss = criterion(qry_emb, pos_emb, torch.tensor([1.0]))
            neg_loss = criterion(qry_emb, neg_emb, torch.tensor([-1.0]))
            tot_loss = pos_loss + neg_loss
            tot_loss.backward()
            optimizer.step()
        
    model_path = UPDATED_MODEL_PATH
    torch.save(two_tower.state_dict(), model_path)
    print(f'Model saved to {model_path}')

