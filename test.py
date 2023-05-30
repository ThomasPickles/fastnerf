import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from render import render_rays, get_points_along_rays, get_points_in_slice, get_voxels_in_slice
from datasets import get_params
from helpers import *
from sampling import get_samples_around_point
from phantom import get_sigma_gt, local_to_world

class IndexedDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx], idx

	def __len__(self):
		return self.data.shape[0]

class SamplesDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0] 

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.data[idx,:]

@torch.no_grad()
def get_ray_sigma(model, points, device):
	model = model.to(device)
	points = points.to(device)
	sigma = model(points)
	return sigma

@torch.no_grad()
def render_slice(model, z, device, resolution, voxel_grid, samples_per_point):
	if voxel_grid:
		vox = get_voxels_in_slice(z, device, resolution)
		points = local_to_world(vox)
	else:
		points = get_points_in_slice(z, device, resolution)
	nb_points = points.shape[0]
	delta = 70. / resolution[0] # for walnut
	points = points.to('cpu') # this might be too big for gpu memory once we add in the samples, so we'll batch them up and then put the batches on the gpu one at a time
	samples = get_samples_around_point(points, delta, samples_per_point) # [nb_points, 3, nb_samples]
	sigma = torch.empty((0,1), device='cpu')
	samples = samples.reshape(nb_points*samples_per_point,3)
	samples_dataset = SamplesDataset(samples)
	samples_loader = DataLoader(samples_dataset, batch_size=100_000)

	for batch in samples_loader:
		batch = batch.to('cuda')
		# print(f"GPU memory allocated after loading tensor (MB): {torch.cuda.memory_allocated()/1024**2:.1f}")
		batch_sigma = model(batch).cpu()
		sigma = torch.cat((sigma,batch_sigma),0)
		# TODO: don't need to keep all samples, can do the
		# averaging here
		del batch
	

	sigma = sigma.reshape(samples_per_point, -1) # [nb_points, samples_per_point]
	sigma = torch.mean(sigma, dim=0)
	return sigma # single channel

@torch.no_grad()	
def render_image(model, frame, **params):
	device = params["device"]
	H = params["H"]
	W = params["W"]
	hf = params["hf"]
	hn = params["hn"]
	nb_bins = params["nb_bins"]
	MAX_BATCH_SIZE = 2500 # out-of-memory if we do any more

	dataset = IndexedDataset(frame)
	data = DataLoader(dataset, batch_size = MAX_BATCH_SIZE)

	img_tensor = torch.zeros_like(frame[...,6]) # single channel
	for batch, idx in data:
		ray_origins = batch[...,:3].squeeze(0).to(device)
		ray_directions = batch[...,3:6].squeeze(0).to(device)
		regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
		img_tensor[idx,...] = regenerated_px_values.cpu()

	return img_tensor


@torch.no_grad()
def test_model(model, dataset, img_index, **render_params):
	frame = dataset[img_index]
	print(frame.shape)

	img_tensor = render_image(model, frame, **render_params)

	gt = frame[...,6].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() # probably not the best measure, we want peak
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	