#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

import json, re
from nerf import FastNerf, Cache
from render import render_rays, get_sigma_values
from convert_data import BlenderDataset
from datasets import get_params
from helpers import *

@torch.no_grad()
def get_ray_alpha(model, dataset, img_index, hn, hf, device, nb_bins, H, W):
	view = dataset[img_index]
	ray_ids = torch.randint(0, H*W, (5,))
	print(f"ray_ids: {ray_ids.shape}")
	ray_origins = view[ray_ids,:3].squeeze(0).to(device)
	print(f"ray_origins: {ray_origins.shape}")
	ray_directions = view[ray_ids,3:6].squeeze(0).to(device)
	sigma, _, delta = get_sigma_values(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
	weights = sigma*delta
	weights = weights[:,:-1].transpose(0,1)
	print(f"weights: {weights.shape}")
	fig, ax = plt.subplots()  # Create a figure containing a single axes.
	ax.plot(weights)  # Plot some data on the axes.
	plt.show()

def render_image(view, **params):
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
	# TODO: only need a single value (sigma) to be returned from render_rays
	view = dataset[img_index]
	img_tensor = render_image(view, **render_params)

	gt = view[...,6:].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() # probably not the best measure, we want peak
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("checkpoint", help="The id of the snapshot")
	# parser.add_argument("--px", type=int, default=40, help="The image height to train on")
	# parser.add_argument("--dataset", default='jaw', choices=['lego','jaw'])
	parser.add_argument("--device", default='cpu', choices=['cuda','cpu'])
	parser.add_argument("--cache", action='store_true', help="Runs the fast rendering algo")
	
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	filename = args.checkpoint
	basename = Path(filename).stem 
	snapshot = re.sub('json','pt',filename) 
	with open(filename, 'r') as jsonfile:
		params_dict = json.load(jsonfile)

	# unpack
	layers = params_dict['model']['layers']
	neurons =  params_dict['model']['hidden_dim']
	embed_dim =  params_dict['model']['embedding_dim']
	epochs = params_dict["epochs"]
	lr = params_dict.get("lr")
	img_size = params_dict["img_size"]
	rendering = params_dict["rendering"]
	samples = params_dict["samples"]
	loss = params_dict["loss"]
	final_training_loss_db = params_dict["final_training_loss_db"]
	curve = params_dict["training_loss"]

	# if epochs < 20:
	# 	print("Not fully trained.  Skipping file...")
	# 	exit()
	
	# if lr == None:
	# 	print("No learning rate.  Skipping file...")
	# 	exit()

	device = args.device
	dataset = 'jaw' # args.dataset

	model = FastNerf(embed_dim, layers, neurons)
	model.load_state_dict(torch.load(snapshot))
	model.eval()
	model.to(device)

	# cache = Cache(model, 2.2, device, 96, 64).to(device)
	# del model

	h = img_size
	c, w, (near,far) = get_params(dataset,h)

	# this is a big dataset, and we can't load it all onto the gpu
	# testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True)).to(device)
	testing_dataset = BlenderDataset(dataset, 'transforms_full_b', split="test", img_wh=(w,h), n_chan=c)
	# data_loader = DataLoader(testing_dataset, batch_size=100)

	for img_index in range(1):
		# NOTE: cache sends many pixel values to zero since it does aggressive masking
		# cache seems to send white pixels to black with the lego.
		if args.cache:
			test(cache, near, far, testing_dataset, device=device, img_index=img_index, nb_bins=samples, H=h, W=w)
		else:
			# get_ray_alpha(model, testing_dataset, img_index, hn=near, hf=far, device=device, nb_bins=samples, H=h, W=w)
			test_loss, imgs = batch_test(model, testing_dataset, img_index, hn=near, hf=far, device=device, nb_bins=samples, H=h, W=w)
			cpu_imgs = [img.data.cpu().numpy().reshape(h, w, 3) for img in imgs]
			text = f"test_loss: {test_loss:.1f}dB, training_loss: {final_training_loss_db}dB\nlr: {lr}, loss function: {loss}, epochs: {epochs}\nlayers: {layers}, neurons: {neurons}, embed_dim: {embed_dim}, img_size: {img_size},\nrendering: {rendering}, samples: {samples}"
			write_imgs((cpu_imgs,curve), f'novel_views/img_{basename}_{img_index}.png', text)