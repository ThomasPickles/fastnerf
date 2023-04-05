import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from render import render_rays, get_points_along_rays, get_points_in_slice
from datasets import get_params
from helpers import *

@torch.no_grad()
def get_ray_alpha(model, dataset, img_index, hn, hf, device, nb_bins, H, W):
	view = dataset[img_index]
	ray_ids = torch.randint(0, H*W, (5,))
	ray_origins = view[ray_ids,:3].squeeze(0).to(device)
	ray_directions = view[ray_ids,3:6].squeeze(0).to(device)
	x, delta = get_points_along_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
	sigma = model(x)
	sigma = sigma.reshape(delta.shape)
	weights = sigma*delta
	weights = weights[:,:-1].transpose(0,1)
	return weights

def render_slice(model, z, device):
	x = get_points_in_slice(z,device)
	sigma = model(x)
	return sigma.expand(-1, 3)


def render_image(model, view, **params):
	img_tensor = torch.zeros_like(view[...,6:])
	n_batches = 5
	device = params["device"]
	H = params["H"]
	W = params["W"]
	hf = params["hf"]
	hn = params["hn"]
	nb_bins = params["nb_bins"]

	assert H % n_batches == 0, 'not yet dealing with unequal batches' 
	batch_size = int(H*W / n_batches)
	for i in range(n_batches):
		i_start, i_end = i*batch_size, (i+1)*batch_size
		batch = view[i_start:i_end,...]
		ray_origins = batch[...,:3].squeeze(0).to(device)
		ray_directions = batch[...,3:6].squeeze(0).to(device)
		# this operation is runs out of memory, hence for loop
		regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
		img_tensor[i_start:i_end,...] = regenerated_px_values
	return img_tensor


@torch.no_grad()
def batch_test(model, dataset, img_index, **render_params):
	view = dataset[img_index]
	img_tensor = render_image(model, view, **render_params)

	gt = view[...,6:].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() # probably not the best measure, we want peak
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	