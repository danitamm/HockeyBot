import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-wkrs', '--num_workers', default=0, type=int,
    					help='Number of workers for dataloaders?')
    parser.add_argument('-bsize', '--batch_size', default=256, type=int,
    					help='Batch size for datlaoders?')
    return parser.parse_args()

class AverageMeter:
	def __init__(self):
		self.sum = 0
		self.count = 0
		self.avg = 0

	def reset(self):
		self.sum = 0
		self.count = 0
		self.avg = 0

	def update(self, val, n=1):
		self.sum += n*val
		self.count += n
		self.avg = self.sum / self.count

	@property
	def val(self):
		return self.avg

def get_accuracies(output, target, topk=(1,5)):
	batch_size = target.shape[0]
	_, preds = torch.topk(output, max(topk), dim=1)
	target = target.unsqueeze(-1).expand(preds.shape)
	compare = preds.eq(target)
	accs = [compare[:,:k,...].sum().float().item()/batch_size for k in topk]
	return accs

class HockeyDataset(Dataset):
	def __init__(self, data_file, input_length):
		sequence_length = input_length + 1

		all_texts = [[int(tok) for tok in line.strip('\n').split()] for line in open(data_file)]
		texts = [text for text in all_texts if len(text) >= sequence_length]
		sequences = []
		for text in texts:
			text_subsequences = [text[i:i+sequence_length] for i in range(len(text)-sequence_length+1)]
			sequences.extend(text_subsequences)

		self.num_tokens = max([token for text in all_texts for token in text])+1
		self.input_length = input_length
		self.sequence_length = sequence_length
		self.sequences = sequences

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		x = np.zeros((self.input_length, self.num_tokens), dtype=np.bool)
		y = np.zeros(self.num_tokens, dtype=np.bool)
		for j, tok in enumerate(self.sequences[idx][:-1]):
			x[j,tok] = 1
		y = self.sequences[idx][-1]
		return x, y

class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()
		self.loss = nn.CrossEntropyLoss()

	def forward(self, preds, labels):
		return self.loss(preds, labels)

class MyLSTM(nn.Module):
	def __init__(self, input_length, lstm_dim, dropout, num_tokens):
		super(MyLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size=num_tokens, hidden_size=lstm_dim, 
			batch_first=True, num_layers=1)
		self.drop = nn.Dropout(p=dropout)
		self.dense = nn.Linear(lstm_dim, num_tokens)
		self.sm = nn.Softmax(dim=1)

	def forward(self, x):
		_, (h, _) = self.lstm(x)
		thing = h[0,...]
		h = h.view(h.shape[1],-1)
		h = self.drop(h)
		d = self.dense(h)
		return self.sm(d)

class HockeyAgent:
	def __init__(self):
		args = get_args()
		self.batch_size = args.batch_size
		self.num_workers = args.num_workers

		self.mode = 'train'
		self.data_file = 'data/answers.txt'
		self.input_length = 5
		self.lstm_dim = 128
		self.dropout = 0.5
		self.num_epochs = 100
		# ---------------------------------------
		self.dataset = HockeyDataset(self.data_file, self.input_length)
		self.num_tokens = self.dataset.num_tokens
		self.loader = DataLoader(self.dataset, batch_size=self.batch_size, 
			num_workers=self.num_workers, shuffle=False)
		self.model = MyLSTM(self.input_length, self.lstm_dim, self.dropout, self.num_tokens)
		self.loss = CrossEntropyLoss()
		
		if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)
		self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.optimizer = Adam(self.model.parameters())

	def run(self):
		if self.mode == 'train':
			self.train()

	def train(self):
		for self.cur_epoch in range(self.num_epochs):
			self.train_one_epoch()
			accuracy = self.validate()
			self.save_checkpoint()

	def train_one_epoch(self):
		self.model.train()
		loss, acc1, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
		for x, y in tqdm(self.loader):
			x = x.float()
			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)

			current_loss = self.loss(output, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()
			loss.update(current_loss.item())

			acc1_cur, acc5_cur = get_accuracies(output, y, topk=(1,5))
			acc1.update(acc1_cur, y.shape[0])
			acc5.update(acc5_cur, y.shape[0])
		print('Training epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - top 1 accuracy: '+str(round(acc1.val,5))+
			' - top 5 accuracy: '+str(round(acc5.val,5)))

	def save_checkpoint(self, filename='checkpoint.pth.tar', 
		best_filename='model_best.pth.tar', is_best=False):
		torch.save(self.model.state_dict(), filename)
		if is_best: torch.save(self.model.state_dict(), filename)

	def load_checkpoint(self, filename='checkpoint.pth.tar'):
		self.model.load_state_dict(torch.load(filename))

	def validate(self):
		self.model.eval()
		loss, acc1, acc5 = AverageMeter(), AverageMeter(), AverageMeter()
		for x, y in tqdm(self.loader):
			x = x.float()
			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)
			current_loss = self.loss(output, y)

			acc1_cur, acc5_cur = get_accuracies(output, y, topk=(1,5))
			acc1.update(acc1_cur, y.shape[0])
			acc5.update(acc5_cur, y.shape[0])
		print('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - top 1 accuracy: '+str(round(acc1.val,5))+
			' - top 5 accuracy: '+str(round(acc5.val,5)))

agent = HockeyAgent()
agent.run()