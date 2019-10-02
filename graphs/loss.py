import torch.nn as nn

class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()
		self.loss = nn.CrossEntropyLoss()

	def forward(self, preds, labels):
		return self.loss(preds, labels)