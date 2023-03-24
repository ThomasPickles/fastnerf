import torch
# import numpy as np
import torch.nn as nn

class FastNerf(nn.Module):
	# NOTE: currently it's a much smaller network, but it's certainly enough to show
	# the shape of the objects.  At least, it works just fine for the lego
	def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim_pos=64, hidden_dim_dir=32, D=8):
		super(FastNerf, self).__init__()

		self.Fpos = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim_pos), nn.ReLU(),
								  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								#   nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								#   nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								#   nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								#   nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
								  nn.Linear(hidden_dim_pos, 3 * D + 1), )

		self.Fdir = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3, hidden_dim_dir), nn.ReLU(),
								  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
								#   nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
								  nn.Linear(hidden_dim_dir, D), )

		self.embedding_dim_pos = embedding_dim_pos
		self.embedding_dim_direction = embedding_dim_direction
		self.D = D

	@staticmethod
	def positional_encoding(x, L):
		out = [x]
		for j in range(L):
			out.append(torch.sin(2 ** j * x))
			out.append(torch.cos(2 ** j * x))
		return torch.cat(out, dim=1)

	def forward(self, o, d):
		sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim_pos))
		sigma = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])  # [batch_size, 1]
		uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))  # [batch_size, 3, D]

		beta = torch.softmax(self.Fdir(self.positional_encoding(d, self.embedding_dim_direction)), -1)
		color = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
		return color, sigma


class Cache(nn.Module):
	def __init__(self, model, scale, device, Np, Nd):
		super(Cache, self).__init__()

		with torch.no_grad():
			# Position
			x, y, z = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Np).to(device),
									  torch.linspace(-scale / 2, scale / 2, Np).to(device),
									  torch.linspace(-scale / 2, scale / 2, Np).to(device)], indexing='ij')
			xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
			sigma_uvw = model.Fpos(model.positional_encoding(xyz, model.embedding_dim_pos))
			self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
			# Direction
			xd, yd = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Nd).to(device),
									 torch.linspace(-scale / 2, scale / 2, Nd).to(device)], indexing='ij')
			xyz_d = torch.cat((xd.reshape(-1, 1), yd.reshape(-1, 1),
							   torch.sqrt((1 - xd ** 2 - yd ** 2).clip(0, 1)).reshape(-1, 1)), dim=1)
			beta = model.Fdir(model.positional_encoding(xyz_d, model.embedding_dim_direction))
			self.beta = beta.reshape((Nd, Nd, -1))

		self.scale = scale
		self.Np = Np
		self.Nd = Nd
		self.D = model.D

	def forward(self, x, d):
		color = torch.zeros_like(x)
		sigma = torch.zeros((x.shape[0], 1), device=x.device)

		# TODO: watch this to understand what the masking is doing
		# https://www.youtube.com/watch?v=1OsRruZmAaM
		mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (x[:, 2].abs() < (self.scale / 2))
		# Position
		idx = (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1)
		sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]]
		# Direction
		idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1)
		beta = torch.softmax(self.beta[idx[:, 0], idx[:, 1]], -1)

		sigma[mask] = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])  # [batch_size, 1]
		uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))  # [batch_size, 3, D]
		color[mask] = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
		return color, sigma
