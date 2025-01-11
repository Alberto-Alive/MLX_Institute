import torch
import random
import sys
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'----------Using device: {device}')

if device == 'cpu':
    sys.exit("Still on CPU, exiting...")

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

# Load the passage vectors
p_vectors_dict = torch.load("./data/all_passages_vectors.pth", map_location=device)
passage_tensor = p_vectors_dict['passages'].to(device)
num_passages = passage_tensor.size(0)
print(f"Number of passages: {num_passages}")
print(f"Passage tensor shape: {passage_tensor.shape}")
print(f"Passage tensor device: {passage_tensor.device}")
print(f"Model device: {next(two_tower.parameters()).device}")


# Process a few sample vectors
for i in range(3):
    input_vector = passage_tensor[i]
    print(f"\nProcessing passage {i}")

    with torch.no_grad():
        embedding = two_tower.doc_tower(input_vector.unsqueeze(0))
    
    print(f"Input vector (first 5 elements): {input_vector[:5]}")
    print(f"Output embedding: {embedding.squeeze(0)}")

if passage_tensor.dim() == 1:
    passage_tensor = passage_tensor.unsqueeze(0)

# Process all passages at once
with torch.no_grad():
    passage_embeddings = two_tower.doc_tower(passage_tensor)

print(f"Passage embeddings shape: {passage_embeddings.shape}")
print("Sample embedding:", passage_embeddings[0])


# Check the embeddings' variability
embedding_std = passage_embeddings.std(dim=0)
print(f"Embedding standard deviation across passages: {embedding_std}")

# Compare embeddings of different passages
print("Embedding of passage 0:", passage_embeddings[0])
print("Embedding of passage 1:", passage_embeddings[1])

# Compute cosine similarity between embeddings
cos = torch.nn.CosineSimilarity(dim=0)
similarity = cos(passage_embeddings[0], passage_embeddings[1])
print(f"Cosine similarity between passage 0 and 1: {similarity.item()}")


# Save the embeddings
torch.save({
    'embeddings': passage_embeddings.cpu()  # Move to CPU if needed
}, './data/passage_embeddings.pth')
print("Passage embeddings saved successfully.")
