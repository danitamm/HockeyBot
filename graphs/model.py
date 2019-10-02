import torch.nn as nn

class MyLSTM(nn.Module):
	def __init__(self, input_length, lstm_dim, num_tokens, dropout=0.5):
		super(MyLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size=num_tokens, hidden_size=lstm_dim, 
			batch_first=True, num_layers=1)
		self.drop = nn.Dropout(p=dropout)
		self.dense = nn.Linear(lstm_dim, num_tokens)
		# self.sm = nn.Softmax(dim=1)

	def forward(self, x):
		self.lstm.flatten_parameters()
		_, (h, _) = self.lstm(x)
		h = h.view(h.shape[1],-1)
		h = self.drop(h)
		d = self.dense(h)
		# d = self.sm(d)
		return d