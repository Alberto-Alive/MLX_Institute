import torch
import torch.nn.functional as F
import math

vocab = {"?": 0, "A": 1, "B": 2, "C": 3, "D":4}
# train = [
#     (["?", "A", "A"], "A"),
#     (["B", "?", "B"], "B"),
#     (["C", "C", "?"], "C")
# ]

train = [
    # Pattern Set 1: Neighbor Dependency
    (["A", "?", "B"], "A"),
    (["C", "B", "?"], "B"),
    (["B", "C", "?"], "C"),

    # Pattern Set 2: Surrounding Context
    (["A", "?", "A"], "A"),
    (["B", "A", "?"], "A"),
    (["C", "?", "B"], "C"),

    # Pattern Set 3: Alternating Pattern
    (["A", "B", "?", "A"], "B"),
    (["C", "A", "C", "?"], "A"),
    (["B", "?", "B", "A"], "A"),

    # Pattern Set 4: Sequential Order with Specific Token Completion
    (["A", "B", "C", "?"], "D"),
    (["?", "B", "C", "D"], "A"),
    (["A", "?", "C", "D"], "B"),

    # Pattern Set 5: Nested or Repetitive Sequences
    (["A", "A", "?", "B", "B"], "A"),
    (["C", "C", "B", "?", "B"], "C"),
    (["B", "A", "B", "?", "A"], "B")
]


torch.manual_seed(29)



class Magic(torch.nn.Module):
    def __init__(self):
        super(Magic, self).__init__()
        self.W_Q = torch.nn.Linear(9,9) # weights matrix for queries
        self.W_K = torch.nn.Linear(9,9) # weights matrix for keys
        self.W_V = torch.nn.Linear(9,9) # weights matrix for values
    def forward(self, inputs):
        Q = self.W_Q(inputs) # embedding matrix times the weights matrix for queries
        K = self.W_K(inputs) # embedding matrix times the weights matrix for keys
        V = self.W_V(inputs) # embedding matrix times the weights matrix for values
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, V)
        return out

class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.max_sequence_length = 128
        self.embedding_dim = 9
        self.emb = torch.nn.Embedding(5, 9) # create an embedding matrix
        # Initialize positional encoding matrix
        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim))
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                if dimension % 2 == 0:
                    pos_encoding_matrix[position, dimension] = math.sin(angle_rate)
                else:
                    pos_encoding_matrix[position, dimension] = math.cos(angle_rate)

        # Register as a buffer to ensure it’s saved with the model but isn’t a trainable parameter
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic() for _ in range(4)]) # create multiple layers of Magic()
        self.vocab = torch.nn.Linear(9, 5) # create a matrix for linear transformation before softmax
    def forward(self, inputs):
        embs = self.emb(inputs) # get the embedding for a certain token
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        # embs = embs + pos_encodings
        for magic in self.magics: embs = magic(embs) 
        logits = self.vocab(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs
    

B = BERT()

optimizer = torch.optim.Adam(B.parameters(), lr=0.0001)

# first training set
# stop_training = False
# for epoch in range(500):
#     count = 0
#     for input, target in train:
#         optimizer.zero_grad()

#         ips = torch.tensor([vocab[w] for w in input]).unsqueeze(0)
        
#         out = B(ips)

#         idx = input.index('?')
#         prd = out[:, idx, :]
#         tgt = torch.tensor([vocab[target]])
#         loss = F.nll_loss(prd, tgt)
#         loss.backward()
#         optimizer.step()

#         predicted_token_id = prd.argmax(dim=-1).item()  # Find the index with the highest log-probability
#         predicted_token = list(vocab.keys())[list(vocab.values()).index(predicted_token_id)]
#         print(f"Predicted token for {target}:", predicted_token)
#         if target == predicted_token:
#             consecutive_correct += 1
#             # Check if all examples have been predicted correctly in a row
#             if consecutive_correct == len(train):
#                 print("Model correctly predicted all targets in a row. Stopping training.")
#                 stop_training = True
#                 break  # Break the inner loop
#         else:
#             consecutive_correct = 0  # Reset counter if prediction is incorrect
#         print("Loss:", loss.item())
#     if stop_training:
#         print(f"Training completed successfully after {epoch + 1} epochs.")
#         break  # Break the outer loop
#     print("^====================================================================^")
   




# second training set
stop_training = False
for epoch in range(10000):
    correct_count = 0  # Track correct predictions within the epoch
    for input_sequence, target in train:
        optimizer.zero_grad()

        # Prepare input sequence and get model output
        ips = torch.tensor([vocab[w] for w in input_sequence]).unsqueeze(0)
        out = B(ips)

        # Get the index of "?" in the sequence and model's predicted output for it
        idx = input_sequence.index('?')
        prd = out[:, idx, :]
        tgt = torch.tensor([vocab[target]])
        loss = F.nll_loss(prd, tgt)
        loss.backward()
        optimizer.step()

        # Get predicted token and check if it matches the target
        predicted_token_id = prd.argmax(dim=-1).item()  # Find the index with the highest log-probability
        predicted_token = list(vocab.keys())[list(vocab.values()).index(predicted_token_id)]
        print(f"Predicted token for {input_sequence} (target: {target}): {predicted_token}")
        
        if predicted_token == target:
            correct_count += 1
        else:
            correct_count = 0  # Reset if there's any mistake in the predictions

        # Stop if we correctly predicted all examples in this epoch
        if correct_count == len(train):
            stop_training = True
            print("Model correctly predicted all targets in this epoch. Stopping training.")
            break

        print("Loss:", loss.item())
        
    if stop_training:
        print(f"Training completed successfully after {epoch + 1} epochs.")
        break  # Exit the outer loop if all predictions were correct
    print("^====================================================================^")
