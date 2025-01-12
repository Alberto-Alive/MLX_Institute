import torch

torch.manual_seed(51)

# Generate random inputs
inputs = torch.randn(5, 9)
print("Initial inputs:\n", inputs)
print("Shape of inputs:", inputs.shape)

# Compute attention scores as a dot product of inputs with its transpose
attn = inputs @ inputs.T
print("\nAttention scores (inputs @ inputs.T):\n", attn)
print("Shape of attention scores:", attn.shape)

# Create a mask with -inf in the upper triangular part, excluding the diagonal
base = torch.full_like(attn, float("-inf"))
mask = torch.triu(base, diagonal=1)
print("\nMask (upper triangular part with -inf):\n", mask)
print("Shape of mask:", mask.shape)

# Apply mask to attention scores
attn = attn + mask
print("\nMasked attention scores:\n", attn)
print("Shape of masked attention scores:", attn.shape)

# Apply softmax to the masked attention scores to get probabilities
probs = attn.softmax(dim=-1)
print("\nSoftmax probabilities of masked attention scores:\n", probs)
print("Shape of softmax probabilities:", probs.shape)
