import torch
import json
import random
import time

torch.manual_seed(42)
random.seed(42)

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

# Initialize model, criterion, and optimizer
two_tower = TwoTower()
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(two_tower.parameters(), lr=0.0001)


# Load the saved triplet vectors
triplet_data = torch.load('triplet_vectors.pth')
qry_tensor = triplet_data['queries']      # Shape: [N, 64]
pos_tensor = triplet_data['positives']    # Shape: [N, 64]
neg_tensor = triplet_data['negatives']    # Shape: [N, 64]

# Iterate over the tensors
# Timing variable outside the loop
start_time = time.time()  # Start timing for the first block

for i in range(len(qry_tensor)):
    # Begin a new timing block every 1000 iterations
    if i % 1000 == 0 and i != 0:
        end_time = time.time()
        print(f'Iteration {i}: Last 1000 iterations took {end_time - start_time:.2f} seconds')
        start_time = time.time()  # Reset start time for the next 1000 iterations
    
    # Main loop code
    qry = qry_tensor[i].unsqueeze(0)  # Shape: [1, 64]
    pos = pos_tensor[i].unsqueeze(0)  # Shape: [1, 64]
    neg = neg_tensor[i].unsqueeze(0)  # Shape: [1, 64]
    optimizer.zero_grad()
    qry_emb, pos_emb = two_tower(qry, pos)
    _, neg_emb = two_tower(qry, neg)
    pos_loss = criterion(qry_emb, pos_emb, torch.tensor([1.0]))
    neg_loss = criterion(qry_emb, neg_emb, torch.tensor([-1.0]))
    tot_loss = pos_loss + neg_loss
    tot_loss.backward()
    optimizer.step()
    
    # Print loss at every 1000th iteration
    if i % 1000 == 0:
        print(f'Iteration {i + 1}: Loss = {tot_loss.item()}')

# Optional final timing for any remaining iterations if not a multiple of 1000
if len(qry_tensor) % 1000 != 0:
    end_time = time.time()
    print(f'Final {len(qry_tensor) % 1000} iterations took {end_time - start_time:.2f} seconds')

model_path = 'two_tower_model_cpu.pth'
torch.save(two_tower.state_dict(), model_path)
print(f'Model saved to {model_path}')

