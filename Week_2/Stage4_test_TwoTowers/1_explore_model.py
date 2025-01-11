import torch
import json
import random
import sys
import time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

# Initialize model
two_tower = TwoTower().to(device)  # Ensure the model architecture matches

# Path to your saved state_dict
model_path = './models/two_tower_model_gpu_bs500_epochs10_lr1e-05_loss0.55.pth'  # Update with actual filename

# Load the state_dict
state_dict = torch.load(model_path, map_location=device)
two_tower.load_state_dict(state_dict)
print("Model state_dict loaded successfully.")

# Set model to evaluation mode
two_tower.eval()

# ------------- Exploring the Model ------------- #

# 1. Print the entire model architecture
print("\n----- Model Architecture -----")
print(two_tower)

# 2. List all layers and their parameters
print("\n----- Model Parameters -----")
for name, param in two_tower.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}")

# 3. Access and print specific layer weights
print("\n----- Specific Layer Weights -----")
# Example: Query Tower - First Fully Connected Layer
print("\nQuery Tower - fc1 Weights:")
print(two_tower.qry_tower.fc1.weight)

print("\nQuery Tower - fc1 Biases:")
print(two_tower.qry_tower.fc1.bias)

# Example: Document Tower - Third Fully Connected Layer
print("\nDocument Tower - fc3 Weights:")
print(two_tower.doc_tower.fc3.weight)

print("\nDocument Tower - fc3 Biases:")
print(two_tower.doc_tower.fc3.bias)

# 4. Summary of Parameters
total_params = sum(p.numel() for p in two_tower.parameters())
trainable_params = sum(p.numel() for p in two_tower.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# 5. (Optional) Visualize Model Graph with TensorBoard
# Uncomment the following section if you wish to visualize the model graph

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/two_tower_model_exploration')
# dummy_query = torch.randn(1, 64).to(device)  # Adjust input size if different
# dummy_doc = torch.randn(1, 64).to(device)
# writer.add_graph(two_tower, (dummy_query, dummy_doc))
# writer.close()
# print("\nTensorBoard graph has been saved to 'runs/two_tower_model_exploration'.")

# 6. (Optional) Inspect Embedding Outputs
# Forward a sample input and inspect the embeddings

# Sample input tensors
sample_query = torch.randn(1, 64).to(device)
sample_doc = torch.randn(1, 64).to(device)

with torch.no_grad():
    qry_emb, doc_emb = two_tower(sample_query, sample_doc)

print("\n----- Sample Embeddings -----")
print("Query Embedding:")
print(qry_emb)

print("\nDocument Embedding:")
print(doc_emb)
