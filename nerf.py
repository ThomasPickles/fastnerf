import torch
import torch.nn as nn
import json

class FastNerf(nn.Module):

	def __init__(self, encoding, network):
		super(FastNerf, self).__init__()

		# Encode the input:
		if encoding["otype"] == "Frequency":
			# input size is 2 * 3 * embed_dim + 3
			# factor of 2 for sin and cos
			# factor of 3 for x,y,z ordinates
			# embed_dim is maximum value of frequency in sin/cos
			# additional 3 is for bias terms in x,y,z when frequency=0
			input_dim = 3 * (2 * encoding["n_frequencies"] + 1)
		else:
			# No encoding is actually just n_frequencies = 0!
			input_dim = 3

		# Set up the MLP:

		hidden_dim = network["neurons_per_layer"]
		
		self.Fpos = nn.Sequential()
		self.Fpos.add_module(f"input", nn.Linear(input_dim, hidden_dim))
		self.Fpos.add_module(f"input_relu", nn.ReLU())

		for i in range(network["n_layers"]):
			self.Fpos.add_module(f"hidden_{i}", nn.Linear(hidden_dim, hidden_dim)) 
			self.Fpos.add_module(f"relu_{i}", nn.ReLU())
								  
		self.Fpos.add_module(f"output", nn.Linear(hidden_dim, 1))

		self.encoding = encoding
		self.network = network

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

		# Lookup is probably negligible compared to query
		# time of network
		if self.encoding["otype"] == "Frequency":
			o = self.positional_encoding(o, self.encoding["n_frequencies"])

		sigma = self.Fpos(o)
		# constrain to be positive
		sigma = torch.nn.functional.softplus(sigma[:, 0][..., None]) 
		return sigma