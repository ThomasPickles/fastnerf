#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import json, re
from nerf import FastNerf, Cache
from render import render_rays
from convert_data import BlenderDataset
from datasets import get_params
from helpers import *

@torch.no_grad()
def batch_test(model, hn, hf, dataset, device, img_index, nb_bins, H, W):
	
	# TODO: only need a single value (sigma) to be returned from render_rays
	img = dataset[img_index]
	img_tensor = torch.zeros_like(img[...,6:])
	n_batches = 5
	assert H % n_batches == 0, 'not yet dealing with unequal batches' 
	batch_size = int(H*W / n_batches)
	# TODO: vectorise this for loop somehow with better batching?
	for i in range(n_batches):
		i_start, i_end = i*batch_size, (i+1)*batch_size
		batch = img[i_start:i_end,...]
		ray_origins = batch[...,:3].squeeze(0).to(device)
		ray_directions = batch[...,3:6].squeeze(0).to(device)
		# this operation is runs out of memory, hence for loop
		regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
		img_tensor[i_start:i_end,...] = regenerated_px_values

	gt = img[...,6:].squeeze(0)
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() # probably not the best measure, we want peak
	print(f"Writing images..., test loss = {linear_to_db(loss):.2f} dB")
	imgs = [img_tensor.data.cpu().numpy().reshape(H, W, 3),gt.data.cpu().numpy().reshape(H, W, 3),diff.data.cpu().numpy().reshape(H, W, 3)]
	write_imgs(imgs, f'novel_views/img_{img_index}.png')

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("checkpoint", help="The id of the snapshot")
	parser.add_argument("--px", type=int, default=40, help="The image height to train on")
	# parser.add_argument("--dataset", default='jaw', choices=['lego','jaw'])
	# parser.add_argument("--device", default='cpu', choices=['cuda','cpu'])
	parser.add_argument("--cache", action='store_true', help="Runs the fast rendering algo")
	
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	filename = args.checkpoint
	snapshot = re.sub('json','pt',filename) 
	with open(filename, 'r') as jsonfile:
		params_dict = json.load(jsonfile)

	# unpack
	layers = params_dict['model']['layers']
	neurons =  params_dict['model']['hidden_dim']
	embed_dim =  params_dict['model']['embedding_dim']
	epochs = params_dict["epochs"]
	img_size = params_dict["img_size"]
	rendering = params_dict["rendering"]
	samples = params_dict["samples"]

	device = 'cpu'
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
	testing_dataset = BlenderDataset(dataset, 'transforms_full', split="test", img_wh=(w,h), n_chan=c)
	# data_loader = DataLoader(testing_dataset, batch_size=100)

	for img_index in range(4):
		# NOTE: cache sends many pixel values to zero since it does aggressive masking
		# cache seems to send white pixels to black with the lego.
		if args.cache:
			test(cache, near, far, testing_dataset, device=device, img_index=img_index, nb_bins=samples, H=h, W=w)
		else:
			batch_test(model, near, far, testing_dataset, device=device, img_index=img_index, nb_bins=samples	, H=h, W=w)