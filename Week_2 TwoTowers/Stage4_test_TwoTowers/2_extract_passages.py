import torch
import json
import faiss
from tqdm import tqdm

# Parameters
EMBEDDING_DIM = 64
VOCAB_PATH = './data/fine_text8_words_to_ids.json'
MODEL_PATH = './models/two_tower_model_gpu_bs500_epochs10_lr1e-05_loss0.55.pth'  # Update with your actual model path
TRIPLET_PATH = './data/cleaned_triplets.json'
TOP_K = 5
UNK_TOKEN = '<UNK>'


# Stage I: List of passages as text
with open(TRIPLET_PATH, 'r') as f:
    cleaned_triplets = json.load(f)
print(f"Total triplets loaded: {len(cleaned_triplets)}")

unique_passages = set()
for triplet in cleaned_triplets:
    _, pos_txt, neg_txt = triplet
    unique_passages.add(pos_txt)
    unique_passages.add(neg_txt)
print(f"Unique passages extracted: {len(unique_passages)}")

with open('./data/all_passages.json', 'w') as f:
    json.dump(list(unique_passages), f)
print("Unique passages saved to ./data/all_passages.json")


















# # Step 1: Load the vocabulary mapping
# with open(VOCAB_PATH, 'r') as f:
#     word_to_idx = json.load(f)
# print(f"Vocabulary size loaded: {len(word_to_idx)}")

# # Step 2: Load the state_dict to get embedding weights
# state_dict = torch.load(MODEL_PATH, map_location='cpu')
# if 'emb.weight' in state_dict:
#     embedding_weights = state_dict['emb.weight']  # Shape: [VOCAB_SIZE, EMBEDDING_DIM]
#     print(f"Embedding weights shape: {embedding_weights.shape}")
# else:
#     raise KeyError("The state_dict does not contain 'emb.weight'.")

# # Step 3: Define the Two-Tower model structure
# class QryTower(torch.nn.Module):
#     def __init__(self):
#         super(QryTower, self).__init__()
#         self.fc1 = torch.nn.Linear(64, 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 32)
#         self.fc4 = torch.nn.Linear(32, 16)

#     def forward(self, x):
#         y = torch.nn.functional.relu(self.fc1(x))
#         y = torch.nn.functional.relu(self.fc2(y))
#         y = torch.nn.functional.relu(self.fc3(y))
#         y = self.fc4(y)
#         return y

# class DocTower(torch.nn.Module):
#     def __init__(self):
#         super(DocTower, self).__init__()
#         self.fc1 = torch.nn.Linear(64, 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 32)
#         self.fc4 = torch.nn.Linear(32, 16)

#     def forward(self, x):
#         y = torch.nn.functional.relu(self.fc1(x))
#         y = torch.nn.functional.relu(self.fc2(y))
#         y = torch.nn.functional.relu(self.fc3(y))
#         y = self.fc4(y)
#         return y

# class TwoTower(torch.nn.Module):
#     def __init__(self):
#         super(TwoTower, self).__init__()
#         self.qry_tower = QryTower()
#         self.doc_tower = DocTower()

#     def forward(self, qry, doc):
#         qry_emb = self.qry_tower(qry)
#         doc_emb = self.doc_tower(doc)
#         return qry_emb, doc_emb

# # Step 4: Initialize and load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Using device: {device}')

# two_tower = TwoTower().to(device)
# two_tower.load_state_dict(state_dict)
# two_tower.eval()
# print("Model loaded successfully.")


# two_tower.doc_emb()
# # Step 5: Define the txt2vec function
# def txt2vec(txt):
#     """
#     Converts a text string into a mean embedding vector.

#     Args:
#         txt (str): The input text.

#     Returns:
#         torch.Tensor: A tensor of shape [EMBEDDING_DIM].
#     """
#     words = txt.lower().split()
#     indices = [word_to_idx.get(word, word_to_idx.get(UNK_TOKEN, 0)) for word in words]
#     if indices:
#         vectors = embedding_weights[indices]  # Shape: [num_words, EMBEDDING_DIM]
#         mean_vector = vectors.mean(dim=0)    # Shape: [EMBEDDING_DIM]
#         return mean_vector
#     else:
#         return torch.zeros(EMBEDDING_DIM)

# # Step 6: Load and process triplets to extract unique passages
# with open(TRIPLET_PATH, 'r') as f:
#     cleaned_triplets = json.load(f)
# print(f"Total triplets loaded: {len(cleaned_triplets)}")

# unique_passages = set()
# for triplet in cleaned_triplets:
#     _, pos_txt, neg_txt = triplet
#     unique_passages.add(pos_txt)
#     unique_passages.add(neg_txt)
# print(f"Unique passages extracted: {len(unique_passages)}")

# with open('./data/all_passages.json', 'w') as f:
#     json.dump(list(unique_passages), f)
# print("Unique passages saved to ./data/all_passages.json")

