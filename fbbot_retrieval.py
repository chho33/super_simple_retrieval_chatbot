from flask import Flask, request
from fb_setting import *
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils import *
import sys
import jieba
jieba.set_dictionary('data/dict_fasttext.txt')
jieba.initialize()
fasttext_model = load_model(fasttext_model_path)

# create a Flask app instance
app = Flask(__name__)

mapping = get_mapping()
with open(target_path, 'r') as f:
    utterance = [row.strip() for row in f.readlines()] 
with open(source_path, 'r') as f:
    context = [row.strip() for row in f.readlines()] 
cs = list(map(lambda x: run_vec_mean(x,model=fasttext_model),context))
cs = np.array(cs)

print('on board....')

# method to reply to a message from the sender
def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    # Post request using the Facebook Graph API v3.1
    resp = requests.post("https://graph.facebook.com/v3.1/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)

# GET request to handle the verification of tokens
@app.route('/', methods=['GET'])
def handle_verification():
    if request.args['hub.verify_token'] == VERIFY_TOKEN:
        return request.args['hub.challenge'], 200
    else:
        return "Invalid verification token", 403

@app.route('/webhook', methods=['GET'])
def handle_verification2():
    if request.args['hub.verify_token'] == VERIFY_TOKEN:
        return request.args['hub.challenge'], 200
    else:
        return "Invalid verification token", 403

# POST request to handle in coming messages then call reply()
@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    sentence= data['entry'][0]['messaging'][0]['message']['text']
    sentence = sentence.strip()
    print('user_input: ',sentence)
    sentence = (' ').join(jieba.lcut(sentence))
    sentence = run_vec_mean(sentence,model=fasttext_model)
    sentence = sentence.reshape(1,-1)
    scores = cosine_similarity(sentence,cs)
    max_indices = list(map(lambda x:np.argmax(x),scores))
    response = [utterance[i] for i in max_indices][0]
    response = ('').join(response.split(' '))
    print('response: ',response)
    reply(sender, response)
    return "ok"

# Run the application.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
