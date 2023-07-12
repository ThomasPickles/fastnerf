import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from render import get_pixel_values, get_points_along_rays, get_points_in_slice, get_voxels_in_slice
from datasets import get_params
from chart_writer import *
from sampling import get_samples_around_point

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
def render_slice(model, dim, device, resolution, voxel_grid, samples_per_point):
	points = get_points_in_slice(dim, device, resolution)
	nb_points = points.shape[0]
	delta = 1. / resolution[0]

	# In general we have too many points to put directly on gpu (res**2 * samples_per_point), so put them on cpu then calculate on gpu in batches
	points = points.to('cpu') 
	samples = get_samples_around_point(points, delta, samples_per_point) # [nb_samples, nb_points, 3]
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
		regenerated_px_values = get_pixel_values(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
		img_tensor[idx,...] = regenerated_px_values.cpu()

	return img_tensor


@torch.no_grad()
def test_model(model, dataset, img_index, **render_params):
	frame = dataset[img_index]

	img_tensor = render_image(model, frame, **render_params)

	gt = frame[...,6].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() 
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	