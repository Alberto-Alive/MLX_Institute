import torch
import torch.nn.functional as F
import more_itertools
import collections
import json
import tqdm
import re
import pandas as pd
import pickle
import wandb


with open('../merged_query_passage.pkl', 'rb') as f:
    loaded_corpus = pickle.load(f)
    print(loaded_corpus[:10])
with open('./text8_words_to_ids.json', 'r') as f:
    text8_words_to_ids = json.load(f)
with open('./text8_ids_to_words.json', 'r') as f:
    text8_ids_to_words = json.load(f)

def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('/',  ' <SLASH> ')
  text = text.replace('\\',  ' <BACK_SLASH> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace(':',  ' <COLON> ')
  text = re.sub(r'[^A-Za-z0-9\s<>]', ' ', text) # remove special characters
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  print(words[:50])
  return words


# Define your SkipGramFoo class again
class SkipGramFoo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs]
        rnd = self.ffw.weight[rand]
        out = torch.mm(ctx, emb.T)
        rnd = torch.mm(rnd, emb.T)
        out = self.sig(out)
        rnd = self.sig(rnd)
        pst = -out.log().mean()
        ngt = -(1 - rnd).log().mean()
        return pst + ngt


def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+2: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  int_to_vocab[1] = '<UNK>'

  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


model_save_path = "./w2v_text8_v4.pth"
args = (len(text8_words_to_ids), 64, 2)  # Same parameters used in training
mFoo_loaded = SkipGramFoo(*args)
mFoo_loaded.load_state_dict(torch.load(model_save_path), strict=False) 


fine_tune_corpus = preprocess(loaded_corpus)  # Apply preprocess on fine-tuning data
bing_words_to_ids, bing_ids_to_words = create_lookup_tables(fine_tune_corpus)


nr_rows = 0
# Merge original and new vocabularies
for word in bing_words_to_ids:
    if word not in text8_words_to_ids:
        new_id = len(text8_words_to_ids)  # Assign new ID
        text8_words_to_ids[word] = new_id
        text8_ids_to_words[new_id] = word
        nr_rows +=1


# # Convert the enhanced corpus to a list of IDs
tokens = [text8_words_to_ids[word] for word in fine_tune_corpus]

# Load the saved model


# Put the model in evaluation mode
mFoo_loaded.train()
new_vocab_size = len(text8_words_to_ids)

print("Number of rows: ", new_vocab_size)
# Print original weights and bias
# print("Original weights:\n", mFoo_loaded.ffw.weight)

# Update embeddings
original_embeddings = mFoo_loaded.emb.weight.data
new_embeddings_row = torch.randn(nr_rows, original_embeddings.size(1))
updated_embeddings = torch.cat([original_embeddings, new_embeddings_row])
new_emb_layer = torch.nn.Embedding(num_embeddings=original_embeddings.size(0) + nr_rows, embedding_dim=original_embeddings.size(1)) # Pytorch defines fixed length embeddings that can't be changed
mFoo_loaded.emb = new_emb_layer
mFoo_loaded.emb.weight.data = updated_embeddings



# Update weights
original_weights = mFoo_loaded.ffw.weight.data
new_weight_row = torch.randn(nr_rows, original_weights.size(1))  # A new row with random values
updated_weights = torch.cat([original_weights, new_weight_row], dim=0)
new_weights_layer = torch.nn.Linear(in_features=original_weights.size(1), out_features=original_weights.size(0) + nr_rows, bias=False)
mFoo_loaded.ffw = new_weights_layer
mFoo_loaded.ffw.weight.data = updated_weights

# mFoo_loaded.emb.weight.requires_grad = False  # Freeze all embeddings

# mFoo_loaded.ffw.weight.requires_grad = False  # Freeze all weights


# mFoo_loaded.emb.weight[-1].requires_grad = True  # Unfreeze the last word's embedding
# mFoo_loaded.ffw.weight[-1].requires_grad = True  # Unfreeze the last word's weight
print("wait", mFoo_loaded.ffw.weight[-1].requires_grad, mFoo_loaded.ffw.weight[-2].requires_grad)

opFoo = torch.optim.Adam(mFoo_loaded.parameters(), lr=0.0003)
model_save_path = "fine_tuned_w2v_text8_v4_query_passage.pth"

orig_last_rw = mFoo_loaded.ffw.weight.data[-1].clone()
# orig_slast_rw = mFoo_loaded.ffw.weight.data[-2]
wandb.init(project='skip-gram', name='mFoo')
for epoch in range(10):
  wins = more_itertools.windowed(tokens[:10000], 3)
  prgs = tqdm.tqdm(enumerate(wins), total=len(tokens[:10000]), desc=f"Epoch {epoch+1}", leave=False)
  current_last_rw = mFoo_loaded.ffw.weight.data[-1].clone()
  if torch.equal(orig_last_rw, current_last_rw):
        print(f"Epoch {epoch+1}: The last row has NOT changed.")
  else:
        print(f"Epoch {epoch+1}: The last row HAS changed.")
  for i, tks in prgs:
    inpt = torch.LongTensor([tks[1]])
    trgs = torch.LongTensor([tks[0], tks[2]])
    rand = torch.randint(0, len(text8_words_to_ids), (2,))
    opFoo.zero_grad()
    loss = mFoo_loaded(inpt, trgs, rand)
    loss.backward()
    with torch.no_grad():
        mFoo_loaded.emb.weight.grad[:-nr_rows] = 0  
        mFoo_loaded.ffw.weight.grad[:-nr_rows] = 0 
    opFoo.step()
    wandb.log({'loss': loss.item()})
torch.save(mFoo_loaded.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
wandb.finish()




