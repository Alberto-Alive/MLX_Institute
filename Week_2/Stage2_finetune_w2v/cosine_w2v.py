import torch
import json

# Load words_to_ids (word to index) and ids_to_words (index to word)
with open('./data/text8_words_to_ids.json', 'r') as f:
    words_to_ids = json.load(f)

with open('./data/text8_ids_to_words.json', 'r') as f:
    ids_to_words = json.load(f)

#FineTune
# with open('./data/fine_text8_words_to_ids.json', 'r') as f:
#     words_to_ids = json.load(f)
# with open('./data/fine_text8_ids_to_words.json', 'r') as f:
#     ids_to_words = json.load(f)

# Define your SkipGramFoo class
class SkipGramFoo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

# Load the saved model
# model_save_path = "./models/gpu_w2v_text8_v4.pth" # ['<PAD>', '<UNK>', 'the', 'of', 'and']
# model_save_path = "./models/ugpu_w2v_text8_v4.pth" # ['manoa', 'humvee', 'serengeti', 'referenda', 'communism']
# model_save_path = "./models/cpu_w2v_text8_v4.pth" # ['<PAD>', '<UNK>', 'the', 'of', 'and']
# model_save_path = "./models/halfbatch1epoch_ugpu_w2v_text8_v4.pth" # ['referenda', 'antecedent', 'manoa', 'humvee', 'cartographic']
# model_save_path = "./models/_w2v_text8_v4.pth" # ['manoa', 'humvee', 'serengeti', 'referenda', 'communism']
# model_save_path = "./models/FineTunedTwoTowerBase.pth" # ['manoa', 'humvee', 'serengeti', 'referenda', 'communism']
model_save_path = "./models/30batch_gpu_w2v_text8_v4.pth" # ['manoa', 'humvee', 'serengeti', 'referenda', 'communism']

args = (len(words_to_ids), 64, 2)  # Same parameters used in training
mFoo_loaded = SkipGramFoo(*args)
mFoo_loaded.load_state_dict(torch.load(model_save_path))

# Put the model in evaluation mode
mFoo_loaded.eval()

# Define the function to get the top 5 most similar words
def get_top_similar_words(input_word, model, words_to_ids, ids_to_words, top_n=5):
    if input_word not in words_to_ids:
        print(f"Word '{input_word}' is not in the vocabulary.")
        return []
    
    # Get the embedding for the input word
    input_token = words_to_ids[input_word]
    with torch.no_grad():
        input_embedding = model.emb(torch.LongTensor([input_token]))  # Shape: [1, embedding_dim]

    # Calculate cosine similarity with all words in the vocabulary
    all_embeddings = model.emb.weight  # Shape: [vocab_size, embedding_dim]
    similarities = torch.nn.functional.cosine_similarity(input_embedding, all_embeddings)

    # Get the indices of the top N most similar words
    top_indices = similarities.argsort(descending=True)[:top_n+1]  # +1 to exclude the word itself

    # Exclude the input word itself from results
    top_similar_indices = [idx for idx in top_indices if idx != input_token][:top_n]

    # Map indices back to words
    top_similar_words = [ids_to_words[str(idx.item())] for idx in top_similar_indices]
    
    return top_similar_words

# Example usage
input_word = "king"
top_similar_words = get_top_similar_words(input_word, mFoo_loaded, words_to_ids, ids_to_words, top_n=5)
print(f"Top 5 words similar to '{input_word}': {top_similar_words}")
