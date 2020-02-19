import time
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compare_temps(distribution, T_1, T_2):
	n = distribution.shape[0] // 20
	s_1 = F.softmax(distribution/T_1, dim=0).data.numpy()
	s_2 = F.softmax(distribution/T_2, dim=0).data.numpy()
	plt.bar(list(range(n)), s_1[:n], 3, alpha=0.5, color='r', 
			label='Temperature = {}'.format(T_1))
	plt.bar(list(range(n)), s_2[:n], 3, alpha=0.5, color='b', 
			label='Temperature = {}'.format(T_2))
	plt.legend()
	plt.xlabel('Word Index')
	plt.ylabel('Probability')
	plt.show()