import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import time, json, uuid
import os
import random

from convert_data import BlenderDataset
from nerf import FastNerf
from datasets import get_params
from helpers import linear_to_db
from test import *
from train import train
from render import get_points_along_rays
import helpers as my
# from phantom import get_sigma_gt

import commentjson as json

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument('config', nargs="?", default="config/small.json", help="Configuration parameters")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	with open(args.config) as f:
		config = json.load(f)

	device = config["hardware"]["train"]
	device_name = torch.cuda.get_device_name(device)

	data_config = config["data"]

	data_name = data_config["dataset"]
	h = data_config["img_size"]

	if h < 100:
		print(f"Warning: low res images can lead to jaggies in training")

	c, w, (near,far) = get_params(data_name, h)

	optim = config["optim"]
	seed = None if optim["random_seed"] else 0
	random.seed(seed)

	run_name = uuid.uuid4().hex[0:10] if config["output"]["hash_naming"] else f"todo_NAMING_CONVENTION"

	training_dataset = BlenderDataset(data_name, data_config["transforms_file"], split="train", img_wh=(w,h), n_chan=c, noise_level=data_config["noise_mean"], noise_sd=data_config["noise_sd"], n_train=data_config["n_images"])
	if optim["pixel_importance_sampling"]:
		pixel_weights = training_dataset.get_pixel_values()
		sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
		data_loader = DataLoader(training_dataset, batch_size=optim["batchsize"], sampler=sampler)
	else:
		data_loader = DataLoader(training_dataset, optim["batchsize"], shuffle=True)
	training_im = training_dataset[:w*h,6:]
	# my.write_img(, f"out/{run_name}-train-img-{args.noise:.0e}-{args.noise_sd:.0f}.png")
	
	model = FastNerf(config["encoding"]["n_frequencies"], config["network"]["n_layers"], config["network"]["neurons_per_layer"]).to(device)

	optim = config["optim"]
	model_optimizer = torch.optim.Adam(model.parameters(), lr=optim["learning_rate"])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=optim["milestones"], gamma=optim["gamma"])
	
	if optim["loss"] == 'L2':
		loss_function = nn.MSELoss()
	else:
		print('Loss not implemented')
		exit()

	now = time.monotonic()
	trained_model, training_loss = train(model, model_optimizer, scheduler, data_loader, nb_epochs=optim["training_epochs"], device=device, hn=near, hf=far, nb_bins=optim["samples_per_ray"], loss_function=loss_function)
	training_time = time.monotonic() - now
	timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")

	snapshot_path = f"checkpoints/{run_name}.pt"
	torch.save(model.state_dict(), snapshot_path)
	print(f"Training complete. Model weights saved as {snapshot_path}")

	has_gt = False # TODO: fix ground truth True if (args.dataset == 'jaw') else False 
	if has_gt:
		phantom = np.load("jaw/jaw_phantom.npy")

	test_device = config["hardware"]["test"]
	trained_model.to(test_device)

	output = config["output"]
	
	if output["images"]:
		# no noise in test data
		testing_dataset = BlenderDataset(data_name, data_config["transforms_file"], split="test", img_wh=(w,h), n_chan=c)
		for img_index in range(3):
			test_loss, imgs = test_model(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=test_device, nb_bins=output["samples_per_ray"], H=h, W=w)
			cpu_imgs = [img.data.cpu().numpy().reshape(h, w) for img in imgs]
			# train_img = training_im.reshape(h, w, 3)
			# cpu_imgs.append(train_img)
			view = testing_dataset[img_index]
			
			NB_RAYS = 5
			ray_ids = torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
			px_vals = my.get_px_values(ray_ids, w) 
		
			ray_origins = view[ray_ids,:3].squeeze(0)
			ray_directions = view[ray_ids,3:6].squeeze(0)

			points, delta = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=output["samples_per_ray"])
		
			# since x are calculated randomly, we need to pass in the same values
			sigma = get_ray_sigma(trained_model, points, device=test_device)
			
			if has_gt:
				sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
				sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
				jitter = 0.005*np.random.rand(NB_RAYS,1)
				sigma_gt = sigma_gt + jitter # add some jitter to distinguish values
			else:
				sigma_gt = None

			sigma = sigma.data.cpu().numpy().reshape(NB_RAYS,-1)
			# text = f"test_loss: {test_loss:.1f}dB, training_loss: {float(final_training_loss_db):.1f}dB, lr: {lr:.2E}, loss function: {loss}, training noise level (sd): {args.noise} ({args.noise*(args.noise_sd/256)})\nepochs: {epochs}, layers: {layers}, neurons: {neurons}, embed_dim: {embed_dim}, training time (h): {training_time/3600:.2f}\nnumber of training images: {args.n_train}, img_size: {img_size}, samples per ray: {samples}, pixel importance sampling: {args.importance_sampling}\n"

			my.write_imgs((cpu_imgs, training_loss, sigma, sigma_gt, px_vals), f'out/{run_name}_loss_{img_index}.png', "todo", show_training_img=False)


	is_voxel_grid = True if (data_name == 'jaw') else False 
	MAX_BRIGHTNESS = 2.5 if (data_name == 'jaw') else 10

	# walnut data still needs to be shifted infinitesimally to left

	if output["slices"]:
		# for idx in range(100):
		for idx in range(50,51):
			z = int(3.3*idx) if is_voxel_grid else idx - 50
			resolution = (output["slice_resolution"], output["slice_resolution"]) # TODO: link to config
			img = render_slice(model=trained_model, z=z, device=test_device, resolution=resolution, voxel_grid=is_voxel_grid, samples_per_point = output["rays_per_pixel"])
			img = img.data.cpu().numpy().reshape(resolution[0], resolution[1])/MAX_BRIGHTNESS
			my.write_img(img, f'out/{run_name}_slice_horizontal.png', verbose=True)
		# no video because just slices
		# sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/slice_{run_name}_%03d.png out/{run_name}_slices_{epochs}_{img_size}_{layers}_{neurons}.mp4"
		# os.system(sys_command)
