import torch
import sentencepiece as spm
import json
import torch.nn.functional as F
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(29)

# Magic and BERT Model Definitions
class Magic(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(Magic, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        self.W_Q = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_K = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_V = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_O = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.size()
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        head_outputs = torch.matmul(attn_probs, V)
        concat_output = head_outputs.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)
        out = self.W_O(concat_output)
        out = self.layer_norm(inputs + out)
        return out

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_heads=8, max_sequence_length=128):
        super(BERT, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.emb = torch.nn.Embedding(vocab_size, embedding_dim)

        pos_encoding_matrix = torch.zeros((self.max_sequence_length, self.embedding_dim))
        for position in range(self.max_sequence_length):
            for dimension in range(self.embedding_dim):
                angle_rate = position / (10000 ** (2 * (dimension // 2) / self.embedding_dim))
                pos_encoding_matrix[position, dimension] = math.sin(angle_rate) if dimension % 2 == 0 else math.cos(angle_rate)
        self.register_buffer("pos_encoding", pos_encoding_matrix)

        self.magics = torch.nn.ModuleList([Magic(embedding_dim=self.embedding_dim, num_heads=self.num_heads) for _ in range(4)])
        self.vocab = torch.nn.Linear(self.embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embs = self.emb(inputs)
        pos_encodings = self.pos_encoding[:inputs.size(1), :]
        embs = embs + pos_encodings

        for magic in self.magics:
            embs = magic(embs)

        logits = self.vocab(embs)
        probs = F.log_softmax(logits, dim=-1)
        return probs

# Load model function
def load_model(filepath, vocab_size, embedding_dim=64, num_heads=8, max_sequence_length=128):
    model = BERT(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, max_sequence_length=max_sequence_length)
    # model.load_state_dict(torch.load(filepath))
    state_dict = torch.load(filepath, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Custom Tokenizer Class for SentencePiece
class CustomTokenizer:
    def __init__(self, sp_model_path, vocab_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def encode(self, text, return_tensors='pt'):
        token_ids = self.sp.encode(text, out_type=int)
        return torch.tensor(token_ids).unsqueeze(0)

    def decode(self, ids):
        tokens = [self.id_to_token[id] for id in ids if id in self.id_to_token]
        return self.sp.decode(tokens)

# Testing Functions
def test_embedding_quality(model, sentence_pairs, tokenizer):
    model.eval()
    for sentence1, sentence2 in sentence_pairs:
        inputs1 = tokenizer.encode(sentence1)
        inputs2 = tokenizer.encode(sentence2)

        with torch.no_grad():
            embedding1 = model(inputs1).mean(dim=1)
            embedding2 = model(inputs2).mean(dim=1)
            # embedding1 = model(inputs1)[:, 0, :]
            # embedding2 = model(inputs2)[:, 0, :]

        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Cosine Similarity between \"{sentence1}\" and \"{sentence2}\": {similarity[0][0]:.4f}")


def test_clustering(model, sentences, tokenizer, n_clusters=5):
    model.eval()
    embeddings = []

    # Get embeddings for each sentence
    for sentence in sentences:
        inputs = tokenizer.encode(sentence)
        with torch.no_grad():
            embedding = model(inputs).mean(dim=1)
            embeddings.append(embedding.squeeze().numpy())
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = kmeans.labels_

    # Print clustering results
    for idx, sentence in enumerate(sentences):
        print(f"Sentence: {sentence} - Cluster: {clusters[idx]}")


def test_visualization(model, sentences, tokenizer, method='pca'):
    model.eval()
    embeddings = []

    # Get embeddings
    for sentence in sentences:
        inputs = tokenizer.encode(sentence)
        with torch.no_grad():
            embedding = model(inputs).mean(dim=1)
            embeddings.append(embedding.squeeze().numpy())
    
    embeddings = np.array(embeddings)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence[:20], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))  # annotate with short text
    plt.title(f"Embedding Visualization using {method.upper()}")
    plt.show()





def test_zero_shot_similarity(model, sentence_pairs, tokenizer):
    model.eval()
    for sentence1, sentence2 in sentence_pairs:
        inputs1 = tokenizer.encode(sentence1)
        inputs2 = tokenizer.encode(sentence2)

        with torch.no_grad():
            embedding1 = model(inputs1).mean(dim=1)
            embedding2 = model(inputs2).mean(dim=1)

        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Zero-Shot Similarity between \"{sentence1}\" and \"{sentence2}\": {similarity[0][0]:.4f}")


# Add test_clustering, test_visualization, test_zero_shot_similarity as per your script...

# Load model and tokenizer
model_filepath = "./models/better_name.pt"
sp_model_path = "./models/wiki_tokenizer.model"
vocab_path = "./data/vocabulary.json"
with open(vocab_path, "r") as f:
    loaded_vocab = json.load(f)
vocab_size = len(loaded_vocab)

model = load_model(model_filepath, vocab_size)
custom_tokenizer = CustomTokenizer(sp_model_path, vocab_path)

# Prepare test data and run tests
# sentence_pairs = [
#     ("The cat sits on the mat.", "A cat is on the mat."),
#     ("The sky is blue.", "The ocean is vast and blue."),
#     ("The cat sits on the mat.", "The dog runs in the yard.")
# ]
# sentences = ["The cat sits on the mat.", "The sky is blue.", "The dog barks loudly.", "The car is fast.", "A tree grows tall."]

# Expanded sentence pairs
sentence_pairs = [
    ("The cat sits on the mat.", "A cat is on the mat."),              # Synonym and paraphrase
    ("The sky is blue.", "The ocean is vast and blue."),               # Similar theme
    ("The cat sits on the mat.", "The dog runs in the yard."),         # Different animals, similar structure
    ("He enjoys reading books.", "She likes to read novels."),         # Synonym/Paraphrase with different subjects
    ("The coffee is too hot.", "The tea is very cold."),               # Opposite meanings
    ("There is a storm coming.", "The weather is clear and sunny."),   # Opposite meanings about weather
    ("The project deadline is tomorrow.", "We finished the work yesterday."),  # Temporal contrast
    ("She baked a cake for the party.", "The cake was delicious at the party."), # Related event but different perspective
    ("I will visit the library today.", "He is going to the bookstore."),        # Similar activity in different places
    ("He loves to swim in the lake.", "He hates swimming in the pool."),          # Similar action, contrasting emotions
    # Dissimilar pairs to test model's differentiation
    ("The sun is shining brightly.", "I am learning to play the guitar."),          # Different topic, nature vs. activity
    ("She baked a cake for the party.", "The car is fast and red."),                # Events vs. objects
    ("He completed his homework on time.", "The weather is clear and sunny."),      # Task completion vs. weather
    ("The dog barks loudly.", "The mountain is covered in snow."),                  # Animals vs. nature
    ("The rocket launches into space.", "The food tastes delicious."),              # Science vs. sensory experience
    ("I will visit the library today.", "The game is very exciting."),              # Places vs. emotions
    ("She enjoys painting landscapes.", "The stock market closed higher today."),   # Art vs. finance
    ("The concert was amazing last night.", "He is preparing for a math exam."),    # Entertainment vs. academics
    ("The coffee is too hot.", "The children are playing outside."),                # Food vs. outdoor activity
    ("A tree grows tall.", "The computer runs complex simulations."),               # Nature vs. technology
    ("The ocean waves are calming.", "She loves playing basketball."),              # Nature vs. sports
    ("The fire alarm rang suddenly.", "The cake was very sweet and soft."),         # Alarm vs. food description
    ("The book was published last year.", "The company achieved record profits."),  # Publishing vs. business
    ("He practices basketball every day.", "The weather forecast predicts snow."),  # Sports vs. weather prediction
    ("I enjoy the silence of the forest.", "The city is bustling with life.")       # Nature vs. urban setting
]


# Expanded list of sentences
sentences = [
    "The cat sits on the mat.",          # Animals
    "The sky is blue.",                  # Nature
    "The dog barks loudly.",             # Animals
    "The car is fast.",                  # Objects and speed
    "A tree grows tall.",                # Nature
    "She reads a book in the library.",  # Actions and locations
    "The weather is sunny today.",       # Weather
    "He completed his homework.",        # Actions
    "The food tastes delicious.",        # Food and taste
    "The city is bustling with life.",   # Urban setting
    "She plays the guitar beautifully.", # Art and music
    "The mountain is covered in snow.",  # Nature
    "The children are playing outside.", # Actions and family
    "The rocket launches into space.",   # Science and technology
    "He practices basketball every day." # Sports
]







test_embedding_quality(model, sentence_pairs, custom_tokenizer)
# Run other tests: test_clustering, test_visualization, test_zero_shot_similarity...
# Example usage continued...

# Run embedding quality test
test_embedding_quality(model, sentence_pairs, custom_tokenizer)

# Run clustering test
test_clustering(model, sentences, custom_tokenizer)

# Run embedding visualization test
test_visualization(model, sentences, custom_tokenizer, method='pca')  # or method='tsne'

# Run zero-shot similarity test
test_zero_shot_similarity(model, sentence_pairs, custom_tokenizer)
