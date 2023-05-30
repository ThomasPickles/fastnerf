import torch
import torch.nn as nn
import json

class FastNerf(nn.Module):
	def toDict(self):
		return {
			"layers": self.layers,
			"hidden_dim": self.hidden_dim,
			"embedding_dim": self.embedding_dim
		}

	def __repr__(self):
		return self.toJson()
					# rep = f'NeRF(layers={self.layers},neurons={self.hidden_dim})'
		# return rep

	def __init__(self, embedding_dim, layers, hidden_dim):
		super(FastNerf, self).__init__()

		# input size is 2 * 3 * embed_dim + 3
		self.Fpos = nn.Sequential()
		
		# input size is 2 * 3 * embed_dim + 3
		# factor of 2 for sin and cos
		# factor of 3 for x,y,z ordinates
		# embed_dim is maximum value of frequency in sin/cos
		# additional 3 is for bias terms in x,y,z when frequency=0
		self.Fpos.add_module(f"input", nn.Linear(embedding_dim * 6 + 3, hidden_dim))
		self.Fpos.add_module(f"input_relu", nn.ReLU())

		for i in range(layers):
			self.Fpos.add_module(f"hidden_{i}", nn.Linear(hidden_dim, hidden_dim)) 
			self.Fpos.add_module(f"relu_{i}", nn.ReLU())
								  
		self.Fpos.add_module(f"output", nn.Linear(hidden_dim, 1))

		# direction dependent elements:
		# self.Fdir = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3, hidden_dim_dir), nn.ReLU(),
		# 						  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
		# 						  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
		# 						  nn.Linear(hidden_dim_dir, D), )

		self.layers = layers
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim

	@staticmethod
	def positional_encoding(x, L):
		out = [x]
		for j in range(L):
			out.append(torch.sin(2 ** j * x))
			out.append(torch.cos(2 ** j * x))
		return torch.cat(out, dim=1)

	def forward(self, o):
		# o: [batchsize * nb_bins,3]
		# d: [batchsize * nb_bins,3]
		sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim))
		# constrain to be positive
		sigma = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None]) 
		return sigma