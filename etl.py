from fastText import load_model
import numpy as np
from utils import get_mapping,create_vocab, create_fasttext_vec

fasttext_model_path = 'data/cc.zh.300.bin'
vocab_path = 'data/vocabulary.txt'
vec_path = 'data/fasttext.npy'

create_vocab()
create_fasttext_vec()
