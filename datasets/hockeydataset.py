import numpy as np

from torch.utils.data import Dataset

class HockeyDataset(Dataset):
	def __init__(self, data_file, input_length):
		sequence_length = input_length + 1

		all_texts = [[int(tok) for tok in line.strip('\n').split()] for line in open(data_file)]
		# texts = [text for text in all_texts if len(text) >= sequence_length]
		# sequences = []
		# for text in texts:
		# 	text_subsequences = [text[i:i+sequence_length] for i in range(len(text)-sequence_length+1)]
		# 	sequences.extend(text_subsequences)
		texts = [text for text in all_texts if len(text) >= 2]
		sequences = []
		for text in texts:
			text_subsequences = []
			for i in range(2, len(text)+1):
				if i < sequence_length:
					subsequence = (sequence_length - i)*[-1] + text[:i]
				else:
					subsequence = text[i-sequence_length:i]
				text_subsequences.append(subsequence)
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
			if tok != -1: x[j,tok] = 1
		y = self.sequences[idx][-1]
		return x, y