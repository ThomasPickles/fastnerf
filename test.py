import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from render import render_rays, get_points_along_rays, get_points_in_slice, get_voxels_in_slice
from datasets import get_params
from helpers import *
from phantom import get_sigma_gt, local_to_world

@torch.no_grad()
def get_ray_sigma(model, points, device):
	model = model.to(device)
	points = points.to(device)
	# CHECK: model and points could be on different devices
	# if we've created trained model on gpu and test model
	# on cpu...
	sigma = model(points)
	return sigma

def render_slice(model, z, device, resolution, voxel_grid = False):
	if voxel_grid:
		vox = get_voxels_in_slice(z, device, resolution)
		points = local_to_world(vox)
	else:
		points = get_points_in_slice(z, device, resolution)
	sigma = model(points)
	return sigma.expand(-1, 3)

class IndexedDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx], idx

	def __len__(self):
		return self.data.shape[0]

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
		regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
		img_tensor[idx,...] = regenerated_px_values.cpu()

	return img_tensor


@torch.no_grad()
def batch_test(model, dataset, img_index, **render_params):
	frame = dataset[img_index]
	print(frame.shape)

	img_tensor = render_image(model, frame, **render_params)

	gt = frame[...,6].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() # probably not the best measure, we want peak
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	