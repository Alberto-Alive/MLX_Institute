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
  sys.exit("Still on cpu, exiting...")
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
two_tower = TwoTower().to(device)  # Move model to GPU
criterion = torch.nn.CosineEmbeddingLoss().to(device)  # Move criterion to GPU
learning_rate = 0.00001
optimizer = torch.optim.Adam(two_tower.parameters(), lr=learning_rate)






class TripletDataset(Dataset):
    def __init__(self, qry_tensor, pos_tensor, neg_tensor):
        self.qry_tensor = qry_tensor
        self.pos_tensor = pos_tensor
        self.neg_tensor = neg_tensor

    def __len__(self):
        return len(self.qry_tensor)

    def __getitem__(self, idx):
        qry = self.qry_tensor[idx]
        pos = self.pos_tensor[idx]
        neg = self.neg_tensor[idx]
        return qry, pos, neg


def collate_fn(batch):
    # Separate the queries, positives, and negatives
    qry_batch, pos_batch, neg_batch = zip(*batch)
    # Pad the sequences (if needed) to the max length in the batch
    qry_batch = pad_sequence(qry_batch, batch_first=True)
    pos_batch = pad_sequence(pos_batch, batch_first=True)
    neg_batch = pad_sequence(neg_batch, batch_first=True)
    return qry_batch, pos_batch, neg_batch




# Load the saved triplet vectors
triplet_data = torch.load('./data/triplet_vectors.pth')
qry_tensor = triplet_data['queries'].to(device)      # Move tensors to GPU
pos_tensor = triplet_data['positives'].to(device)
neg_tensor = triplet_data['negatives'].to(device)




# Initialize the dataset
dataset = TripletDataset(qry_tensor, pos_tensor, neg_tensor)

print(len(qry_tensor),len(pos_tensor),len(neg_tensor) )

# Initialize the DataLoader
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



# Timing variable outside the loop
num_epochs = 1
start_time = time.time()  # Start timing for the first block
for epoch in range(num_epochs):  # If you want multiple epochs
    for i, (qry_batch, pos_batch, neg_batch) in enumerate(dataloader):
        # print(qry_batch.shape)
        if i % 1000 == 0 and i != 0:
          end_time = time.time()
          print(f'Iteration {i}: Last 1000 iterations took {end_time - start_time:.2f} seconds')
          start_time = time.time() 

        optimizer.zero_grad()
        
        # Forward pass
        qry_emb, pos_emb = two_tower(qry_batch, pos_batch)
        _, neg_emb = two_tower(qry_batch, neg_batch)
        
        # Calculate loss
        pos_loss = criterion(qry_emb, pos_emb, torch.ones(qry_batch.size(0), device=device))
        neg_loss = criterion(qry_emb, neg_emb, -torch.ones(qry_batch.size(0), device=device))
        batch_loss = pos_loss + neg_loss

        # Backward pass and optimize
        batch_loss.backward()
        optimizer.step()

        # Print progress
        if i % 100 == 0:
            print(f'Batch {i}: Loss = {batch_loss.item()}')
    if len(qry_tensor) % 1000 != 0:
      end_time = time.time()
      print(f'Final {len(qry_tensor) % 1000} iterations took {end_time - start_time:.2f} seconds')


# Save the model
model_path = f'./models/two_tower_model_gpu_bs{batch_size}_epochs{num_epochs}_lr{learning_rate}_loss{batch_loss.item():.2f}.pth'
torch.save(two_tower.state_dict(), model_path)
print(f'Model saved to {model_path}')

