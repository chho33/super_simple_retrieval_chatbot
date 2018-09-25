from fastText import load_model
import numpy as np
import csv
import re

fasttext_model_path = 'data/cc.zh.300.bin'
vocab_path = 'data/vocabulary.txt'
vec_path = 'data/fasttext.npy'
train_path = 'data/train.csv'
valid_path = 'data/valid.csv'
test_path = 'data/test.csv'
source_path = 'data/source'
target_path = 'data/target'

def get_mapping(vocab_path=vocab_path,vec_path=vec_path):
    with open(vocab_path,'r') as f:
        txt = [row.strip() for row in f.readlines()]
    vec = np.load(vec_path)
    mapping = {}
    for t,v in zip(txt,vec):
        mapping[t]=v
    return mapping

def create_fasttext_vec(model_path=fasttext_model_path,mapping=vocab_path,output_file=vec_path,t2s=False):
    model = load_model(model_path)
    text = []
    with open(mapping, 'r') as f:
        for row in f.readlines():
            row = row.strip()
            if t2s:
                row = tradition2simple(row)
            vec = model.get_word_vector(row)
            text.append(vec)
    text = np.array(text)
    np.save(output_file,text)

def create_vocab(csv_files=[source_path,target_path],vocab_path=vocab_path):
    if not isinstance(csv_files,list):
        csv_files = [csv_files]
    vocabs = []
    for c in csv_files:
        with open(c,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0: continue
                #words = row[0].split(' ') + row[1].split(' ')
                words = row[0].split(' ')
                for w in words:
                    if w not in vocabs:
                        vocabs.append(w)
    with open(vocab_path,'w') as f:
        f.write(('\n').join(vocabs))

def map_vec_mean(txt,mapping):
    txt = re.sub(' ?__eou__','',txt) 
    if len(txt) == 0: return np.zeros(300)
    txts = txt.split(' ')
    vecs = list(map(lambda x:mapping[x],txts))
    vecs = np.mean(vecs,axis=0)
    return vecs

def run_vec_mean(txt,model):
    txt = re.sub(' ?__eou__','',txt) 
    if len(txt) == 0: return np.zeros(300)
    txts = txt.split(' ')
    vecs = list(map(lambda x:model.get_word_vector(x),txts))
    vecs = np.mean(vecs,axis=0)
    return vecs
