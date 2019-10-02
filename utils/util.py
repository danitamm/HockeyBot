import argparse
import numpy as np
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-wkrs', '--num_workers', default=0, type=int,
    					help='Number of workers for dataloaders?')
    parser.add_argument('-bsize', '--batch_size', default=256, type=int,
    					help='Batch size for dataloaders?')
    parser.add_argument('-runstrt', '--running_start', default=False,
    					type=bool, help='Start from checkpoint?')
    parser.add_argument('-epochs', '--num_epochs', default=500,
    					type=int, help='Number of epochs?')
    return parser.parse_args()

def get_accuracies(output, target, topk=(1,5)):
	batch_size = target.shape[0]
	_, preds = torch.topk(output, max(topk), dim=1)
	target = target.unsqueeze(-1).expand(preds.shape)
	compare = preds.eq(target)
	accs = [compare[:,:k,...].sum().float().item()/batch_size for k in topk]
	return accs
	
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