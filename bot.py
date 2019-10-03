import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

print('Hello, I\'m the HockeyBot. I\'ve learned from the best and know all there is'+ 
	' to know about handling sports reporters. Please enter the beginning of your'+
	' response, and I\'ll gladly finish it for you.\n')

while True:
	usr_input = input('Please enter the beginning of your response, or adjust'+ 
					' our strategy by entering \'help\'.\n')
	if usr_input == 'help':
		max_sentences_raw = input('We\'re currently giving '+str(max_sentences)+
			' sentence response. Enter your desired number of sentences'+
			' to change this, or hit enter to keep it the same\n')
		if max_sentences_raw and max_sentences_raw.isdigit(): max_sentences = int(max_sentences_raw)
		time.sleep(0.1)
		temperature_raw = input('We\'re currently giving answers with a freedom of '+
			str(temperature)+', where 0 is entirely predictable and anything above 3'+ 
			' becomes incoherent. Enter your desired nonnegative freedom factor,'+ 
			' or hit enter to keep it the same.\n')
		if temperature_raw:
			try: temperature = float(temperature_raw)
			except: pass
		continue
	results = get_prediction(usr_input)
	print('\n' + results + '\n') 