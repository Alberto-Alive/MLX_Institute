import json
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def create_random_index_groups(dataset_size, group_size=4, num_groups=15000, save_path='./data/index_groups.json'):
    index_groups = []
    for _ in range(num_groups):
        group = random.sample(range(dataset_size), group_size)
        index_groups.append(group)
    
    # Save groups to file
    with open(save_path, 'w') as f:
        json.dump(index_groups, f)

# Assuming MNIST has 60,000 images in the training set
create_random_index_groups(dataset_size=60000)




class ImageToEmbedding(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ImageToEmbedding, self).__init__()
        self.flatten = nn.Flatten()  # Flatten 28x28 into 784
        self.linear = nn.Linear(784, embedding_dim)  # Map 784 to 64

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.linear(x)    # Linear layer to create 64-dim embedding
        return x



class MNISTGroupedDataset(torch.utils.data.Dataset):
    def __init__(self, index_groups_file, data_path='./data', train=True):
        # Load index groups
        with open(index_groups_file, 'r') as f:
            self.index_groups = json.load(f)
        
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_dataset = datasets.MNIST(root=data_path, train=train, download=False, transform=transform)
    
    def __len__(self):
        return len(self.index_groups)
    
    def __getitem__(self, idx):
        group_indexes = self.index_groups[idx]
        images, labels = [], []
        
        # Retrieve images and labels based on group indexes
        for index in group_indexes:
            image, label = self.mnist_dataset[index]
            images.append(image)
            labels.append(label)
        
        # Stack images along a new dimension for grouped batch
        return torch.stack(images), torch.tensor(labels)

# Example usage:
grouped_dataset = MNISTGroupedDataset(index_groups_file='./data/index_groups.json')
data_loader = torch.utils.data.DataLoader(grouped_dataset, batch_size=50, shuffle=True, drop_last=False)

for batch_idx, (images, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Images shape: {images.shape}")  # Expected [50, 4, 1, 28, 28] for full batches
    print(f"Labels shape: {labels.shape}")  # Expected [50, 4] for full batches



# Example usage
model = ImageToEmbedding(embedding_dim=64)
sample_image = torch.randn(1, 28, 28)  # Simulate a 28x28 image
embedding = model(sample_image)         # Get 64-dimensional embedding
print(embedding.shape)  # Should print torch.Size([64])


# images, labels = grouped_dataset[0]  # This returns a tuple (images, labels)

# for image in images[:10]: 
#     print("image: ", image) # Iterate over the images in the returned tuple
#     img = transforms.ToPILImage()(image)  # Convert to PIL Image
#     print("Label: ", labels)  # Print the labels for the current group
#     img.show()  # Display the image
    


