import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request

from graphs.model import MyLSTM

def load_checkpoint(model):
	try:
		model.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
		print('Checkpoint loaded as is.\n')
	except:
		# original saved file with DataParallel
		state_dict = torch.load(checkpoint_file, map_location='cpu')
		# create new OrderedDict that does not contain `module.`
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
		    name = k[7:] # remove `module.`
		    new_state_dict[name] = v
		# load params
		model.load_state_dict(new_state_dict)
		print('Checkpoint saved in parallel, converted to fit.\n')

class Converter():
	def __init__(self, token_file, input_length):
		with open(token_file) as f: lines = f.readlines()
		word_to_tok, tok_to_word = {}, {}
		for line in lines:
			word, tok = line.split()
			tok = int(tok)
			word_to_tok[word] = tok
			tok_to_word[tok] = word

		self.num_tokens = len(tok_to_word)
		self.word_to_tok = word_to_tok
		self.tok_to_word = tok_to_word
		self.input_length = input_length

	def to_tok(self, word):
		return -1 if word not in self.word_to_tok else self.word_to_tok[word]

	def to_word(self, tok):
		return '' if tok not in self.tok_to_word else self.tok_to_word[tok]

	def store_tokenized_raw_input(self, usr_input):
		self.words = usr_input.split()
		recent_words = self.words[-self.input_length:]
		seq = [self.to_tok(word) for word in recent_words]
		if len(seq) < self.input_length: seq = (self.input_length-len(seq))*[-1] + seq
		self.sequence = seq

	def seq_to_tensor(self):
		x = torch.zeros((self.input_length, self.num_tokens), dtype=torch.float)
		for i, tok in enumerate(self.sequence):
			if tok != -1: x[i,tok] = 1
		x = x.unsqueeze(0)
		return x

	def add_to_sequence(self, tok):
		self.words.append(self.to_word(tok))
		self.sequence.append(tok)
		self.sequence = self.sequence[-self.input_length:]

	def get_words(self, pretty=True):
		if not pretty: return ' '.join(self.words)
		sentences = self.words.copy()
		sentences[0] = sentences[0].capitalize()
		for i, word in enumerate(sentences[:-1]):
			if word == '.': sentences[i+1] = sentences[i+1].capitalize()
		sentences = ' '.join(sentences)
		return sentences.replace(' .', '.')

def draw_from_output(output, T=1):
	eps = 10**-16 # avoid dividing by 0
	dist = output.data.view(-1)/(T+eps)
	dist = F.log_softmax(dist, dim=0).data
	dist = dist.exp()
	tok = torch.multinomial(dist,1)[0].item()
	return tok

def get_prediction(usr_input):
	converter.store_tokenized_raw_input(usr_input)
	sentence_count = 0
	while sentence_count < max_sentences:
		x = converter.seq_to_tensor()
		output = model(x)
		tok = draw_from_output(output, T=temperature)
		converter.add_to_sequence(tok)
		word = converter.to_word(tok)
		sentence_count += word == '.'

	return converter.get_words()


token_file = 'data/tokens.txt'
checkpoint_file = 'checkpoint600pad.pth.tar'
input_length = 5
lstm_dim = 128
temperature = 0.75
max_sentences = 5
converter = Converter(token_file, input_length)
model = MyLSTM(input_length, lstm_dim, converter.num_tokens)
model.eval()
load_checkpoint(model)

# usr_input = ' '
# results = get_prediction(usr_input)
# print(results)

FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
PAGE_ACCESS_TOKEN = 'EAAG6g1Qk10kBAMJHVMz3hIP8BNtZCWSRm7X1v1ZBFzdiuVdhdRM7cRkLN6B3BfXMKaob66R0XVVPTZCDYuvs6TMbKTDh5hZAwWjlmeyFCRNkQhaeZCiA4VFZCKOgM4ltfDJ25qot2ulJty8Evt81wvtZCUkMqf6l4EF1lrwDUZCHSGKwbx4xoX1ZC'
VERIFY_TOKEN = 'paiiitEeiCkVvVr8sybZanx7bvWhIJ6XMelzmH9NnyM'

app = Flask(__name__)

def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {'message':{'text': text}, 'recipient': {'id': recipient_id}, 
    			'notification_type': 'regular'}
    auth = {'access_token': PAGE_ACCESS_TOKEN}
    response = requests.post(FB_API_URL, params=auth, json=payload)
    return response.json()

def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = get_prediction(message)
    send_message(sender, response)


def is_user_message(message):
    """Check if the message is a message from the user"""
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))

@app.route('/webhook', methods=['GET','POST'])
def listen():
	if request.method == 'GET':
		return verify_webhook(request)

	if request.method == 'POST':
		file = request.json
		event = file['entry'][0]['messaging']
		for x in event:
			if is_user_message(x):
				text = x['message']['text']
				sender_id = x['sender']['id']
				respond(sender_id, text)
		return 'ok'

# @app.route("/webhook")
# def listen():
#     """This is the main function flask uses to 
#     listen at the `/webhook` endpoint"""
#     if request.method == 'GET':
#         return verify_webhook(request)

#     if request.method == 'POST':
#         payload = request.json
#         event = payload['entry'][0]['messaging']
#         for x in event:
#             if is_user_message(x):
#                 text = x['message']['text']
#                 sender_id = x['sender']['id']
#                 respond(sender_id, text)

#         return "ok"