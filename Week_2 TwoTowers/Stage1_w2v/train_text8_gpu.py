#
#
#
import pickle
import tqdm
import collections
import more_itertools
import wandb
import torch
import re
import json
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

#
#
#
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda version?",torch.version.cuda)  
print("Is my gpu compatible?",torch.cuda.is_available()) 
#
#
#

with open('./data/text8') as f: text8: str = f.read()
print("Data was successfully opened")

#
#
#
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
  print(words[:10])
  return words


#
#
#
corpus: list[str] = preprocess(text8)
with open('./data/preprocessed_text8_corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {0: '<PAD>', 1: '<UNK>'}
  int_to_vocab.update({ii + 2: word for ii, word in enumerate(vocab)})

  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


#
#
#
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]

# Save words_to_ids (word to index)
with open('./data/text8_words_to_ids.json', 'w') as f:
    json.dump(words_to_ids, f)

# Save ids_to_words (index to word)
with open('./data/text8_ids_to_words.json', 'w') as f:
    json.dump(ids_to_words, f)

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
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt

#
#
#
args = (len(words_to_ids), 64, 2)
mFoo = SkipGramFoo(*args).to(device)
print("My device is: ", next(mFoo.parameters()).device)

#
#
#

print('mFoo', sum(p.numel() for p in mFoo.parameters()))


opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.001)


windows = list(more_itertools.windowed(tokens, 3))
inputs = [w[1] for w in windows]
targets = [[w[0], w[2]] for w in windows]
input_tensor = torch.LongTensor(inputs)
target_tensor = torch.LongTensor(targets)
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)


model_save_path = "./models/30batch_gpu_w2v_text8_v4.pth"

wandb.init(project='skip-gram', name='mFoo')
mFoo.to(device)
for epoch in range(30):
  start = time.time() 
  prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
  for inpt, trgs in prgs:
    inpt, trgs = inpt.to(device), trgs.to(device)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(device)
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})
  end = time.time() 
  print(f"Epoch {epoch+1} took {end - start} seconds")
torch.save(mFoo.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")
wandb.finish()

