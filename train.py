import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from graphs.loss import CrossEntropyLoss
from graphs.model import MyLSTM
from datasets.hockeydataset import HockeyDataset
from utils.util import get_args, get_accuracies, AverageMeter

class HockeyAgent:
	def __init__(self):
		args = get_args()
		self.batch_size = args.batch_size
		self.num_workers = args.num_workers
		self.running_start = args.running_start
		self.num_epochs = args.num_epochs

		self.mode = 'train'
		self.data_file = 'data/answers.txt'
		self.input_length = 5
		self.lstm_dim = 128
		self.dropout = 0.5
		# ---------------------------------------
		self.dataset = HockeyDataset(self.data_file, self.input_length)
		self.num_tokens = self.dataset.num_tokens
		self.loader = DataLoader(self.dataset, batch_size=self.batch_size, 
			num_workers=self.num_workers, shuffle=True)
		self.model = MyLSTM(
			self.input_length, self.lstm_dim, 
			self.num_tokens, dropout=self.dropout)
		self.loss = CrossEntropyLoss()

		if torch.cuda.device_count() > 1:
			self.model = nn.DataParallel(self.model)
		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')
		else:
			self.device = torch.device('cpu')
		self.model.to(self.device)
		self.optimizer = Adam(self.model.parameters())

		if self.running_start: self.load_checkpoint()

	def run(self):
		if self.mode == 'train':
			self.train()

	def train(self):
		for self.cur_epoch in range(self.num_epochs):
			self.train_one_epoch()
			# accuracy = self.validate()
			self.save_checkpoint()

	def train_one_epoch(self):
		self.model.train()
		loss = AverageMeter()
		acc1 = AverageMeter()
		acc5 = AverageMeter()
		acc10 = AverageMeter()
		for x, y in tqdm(self.loader):
			x = x.float()
			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)

			current_loss = self.loss(output, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()

			acc1_cur, acc5_cur, acc10_cur = get_accuracies(
				output, y, topk=(1,5,10))
			loss.update(current_loss.item())
			acc1.update(acc1_cur, y.shape[0])
			acc5.update(acc5_cur, y.shape[0])
			acc10.update(acc10_cur, y.shape[0])
		print('Training epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,6))+' - top 1 accuracy: '
			+str(round(acc1.val,6))+' - top 5 accuracy: '
			+str(round(acc5.val,6))+' - top 10 accuracy: '
			+str(round(acc10.val,6)))

	def save_checkpoint(self, filename='checkpoint.pth.tar', 
		best_filename='model_best.pth.tar', is_best=False):
		torch.save(self.model.state_dict(), filename)
		if is_best: torch.save(self.model.state_dict(), filename)

	def load_checkpoint(self, filename='checkpoint.pth.tar'):
		filename = 'checkpoint600pad.pth.tar'
		self.cur_epoch = 600
		try:
			self.model.load_state_dict(
				torch.load(filename, map_location=self.device))
			print('Checkpoint loaded as is.')
		except:
			# original saved file with DataParallel
			state_dict = torch.load(filename, map_location=self.device)
			# create new OrderedDict that does not contain `module.`
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
			    name = k[7:] # remove `module.`
			    new_state_dict[name] = v
			# load params
			self.model.load_state_dict(new_state_dict)
			print('Checkpoint saved in parallel,'
				  ' converted for use without parallelization.')

	def validate(self):
		self.model.eval()
		loss = AverageMeter()
		acc1 = AverageMeter()
		acc5 = AverageMeter()
		acc10 = AverageMeter()
		for x, y in tqdm(self.loader):
			x = x.float()
			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)
			current_loss = self.loss(output, y)

			acc1_cur, acc5_cur, acc10_cur = get_accuracies(
				output, y, topk=(1,5,10))
			acc1.update(acc1_cur, y.shape[0])
			acc5.update(acc5_cur, y.shape[0])
			acc10.update(acc10_cur, y.shape[0])
		print('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,6))+' - top 1 accuracy: '
			+str(round(acc1.val,6))+' - top 5 accuracy: '
			+str(round(acc5.val,6))+' - top 10 accuracy: '
			+str(round(acc10.val,6)))

agent = HockeyAgent()
# agent.run()
agent.validate()

'''
Best results:
loss: 2.55
top 1: 0.434
top 5: 0.679
top 10: 0.768
'''