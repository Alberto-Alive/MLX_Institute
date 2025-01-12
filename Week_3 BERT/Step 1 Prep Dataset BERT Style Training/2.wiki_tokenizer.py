import sentencepiece as spm

spm.SentencePieceTrainer.train(input='data/wiki_string.txt', model_prefix='models/wiki_tokenizer', vocab_size=16000, character_coverage=1, model_type='unigram')
