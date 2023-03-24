#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nerf import FastNerf, Cache
from render import render_rays
from convert_data import BlenderDataset
from datasets import get_params

@torch.no_grad()
def test(model, hn, hf, dataset, device='cpu', img_index=0, nb_bins=192, H=400, W=400):
	# prefer this on cpu, since otherwise we're loading a lot of data onto the gpu!
	ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
	ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
	gt = dataset[img_index * H * W: (img_index + 1) * H * W, 6:]
	regenerated_px_values = render_rays(model, ray_origins.to(device), ray_directions.to(device), hn=hn, hf=hf,
										nb_bins=nb_bins)

	img_tensor = regenerated_px_values.data.cpu().numpy().reshape(H, W, 3)
	print(f"Shape is {img_tensor.shape}")
	# print(f"middle row is {img_tensor[int(H/2),:,:]}")
	plt.figure()
	plt.imshow(img_tensor.clip(0, 1))
	plt.axis('off')
	plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
	plt.close()

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("--epochs", type=int, default=8, help="The number of training epochs to run")
	parser.add_argument("--px", type=int, default=40, help="The image height to train on")
	parser.add_argument("--dataset", default='lego', choices=['lego','jaw'])
	parser.add_argument("--device", default='cpu', choices=['cuda','cpu'])
	parser.add_argument("--snapshot", default='snapshot.pt', help="The name of the snapshot")
	parser.add_argument("--cache", action='store_true', help="Runs the fast rendering algo")

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	device = args.device
	dataset = args.dataset
	px = args.px
	# testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

	model = FastNerf()
	model.load_state_dict(torch.load(args.snapshot))
	model.eval()
	model.to(device)

	cache = Cache(model, 2.2, device, 96, 64).to(device)
	# del model

	h = args.px
	c, w = get_params(args.dataset,h)

	# this is a big dataset, and we can't load it all onto the gpu
	# testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True)).to(device)
	testing_dataset = BlenderDataset(dataset, split="test", img_wh=(w,h), n_chan=c)

	for img_index in range(4):
		# NOTE: cache sends many pixel values to zero since it does aggressive masking
		if args.cache:
			test(cache, 2., 6., testing_dataset, device=device, img_index=img_index, nb_bins=192, H=h, W=w)
		else:
			test(model, 2., 6., testing_dataset, device=device, img_index=img_index, nb_bins=192, H=h, W=w)