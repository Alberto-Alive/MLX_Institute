#

import pandas as pd
from timeit import default_timer as timer


wiki_df = pd.read_parquet('./data/wikitext-2-v1.parquet')
print(wiki_df.columns)
print(wiki_df.head())
wiki_string = wiki_df['text'].str.cat(sep=' ')

print(type(wiki_string))

with open('./data/wiki_string.txt', 'w', encoding='utf-8')as file:
    file.write(wiki_string)



