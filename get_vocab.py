import sys
sys.path.append('protein_mdm')
from data.vocabulary import FragmentVocab
vocab = FragmentVocab()
print(f"Vocab size: {vocab.vocab_size}")
print(vocab.idx_to_token)
