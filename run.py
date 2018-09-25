from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import *
import sys
import jieba
jieba.set_dictionary('data/dict_fasttext.txt')
jieba.initialize()
fasttext_model = load_model(fasttext_model_path)

mapping = get_mapping()
'''
df_train = pd.read_csv(train_path) 
df_train = df_train.query('Label==1')
df_train = df_train.drop('Label',axis=1)
df_valid = pd.read_csv(valid_path) 
df_valid = df_valid[["Context","Ground Truth Utterance"]]
df_valid.columns = ["Context","Utterance"]
df_test = pd.read_csv(test_path) 
df_test = df_test[["Context","Ground Truth Utterance"]]
df_test.columns = ["Context","Utterance"]
df = pd.concat([df_train,df_valid,df_test],axis=0)

utterance = df['Utterance']
context = df.Context
'''
with open(target_path, 'r') as f:
    utterance = [row.strip() for row in f.readlines()] 
with open(source_path, 'r') as f:
    context = [row.strip() for row in f.readlines()] 


#cs = context.apply(lambda x: map_vec_mean(x,mapping=mapping))
#cs = np.array([c for c in cs.values])
#cs = np.array([c for c in cs.values])
#cs = list(map(lambda x: map_vec_mean(x,mapping=mapping),context))
cs = list(map(lambda x: run_vec_mean(x,model=fasttext_model),context))
cs = np.array(cs)
sys.stdout.write("Input sentence: ")
sys.stdout.flush()
user_input = sys.stdin.readline()
while(user_input):
    user_input = user_input.strip()
    user_input = (' ').join(jieba.lcut(user_input))
    user_input = run_vec_mean(user_input,model=fasttext_model)
    user_input = user_input.reshape(1,-1)
    scores = cosine_similarity(user_input,cs)
    max_indices = list(map(lambda x:np.argmax(x),scores))
    #response = [utterance.iloc[i] for i in max_indices][0]
    response = [utterance[i] for i in max_indices][0]
    response = ('').join(response.split(' '))
    response = re.sub(' ?__eou__','',response) 
    print('Response: ',response)
    sys.stdout.write("Input sentence: ")
    sys.stdout.flush()
    user_input = sys.stdin.readline()
