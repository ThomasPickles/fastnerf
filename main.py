import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import time, json, uuid
import os
import random
import csv
try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()
	
from convert_data import BlenderDataset
from nerf import FastNerf
from datasets import get_params
from test import *
from render import get_points_along_rays, render_rays
import helpers as my
# from phantom import get_sigma_gt

import commentjson as json

def write_slices(model, device, epoch, sub_epoch, output, prefix):
		MAX_BRIGHTNESS = 10.
		if output["slices"]:
			resolution = (output["slice_resolution"], output["slice_resolution"])
			for axis, name in enumerate(['x','y','z']):
				img = render_slice(model=model, dim=axis, device=device, resolution=resolution, voxel_grid=False, samples_per_point = output["rays_per_pixel"])
				img = img.data.cpu().numpy().reshape(resolution[0], resolution[1])/MAX_BRIGHTNESS
				my.write_img(img, f'{prefix}_{name}_{epoch:04}_{sub_epoch:04}.png', verbose=True)
			# no video because just slices
			# sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/slice_{run_name}_%03d.png out/{run_name}_slices_{epochs}_{img_size}_{layers}_{neurons}.mp4"
			# os.system(sys_command)


def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument('config', nargs="?", default="config/test_config.json", help="Configuration parameters")
	parser.add_argument("--load_checkpoint", default='', help="Load uuid and bypass training")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	with open(args.config) as f:
		config = json.load(f)

	if config["output"]["path"] == "as_config":
		out_dir = os.path.splitext(args.config)[0]
		try:
			os.mkdir(out_dir)
		except FileExistsError:
			print(f"Overwriting previous results...")
	else:
		out_dir = 'tmp'

	device = config["hardware"]["train"]
	device_name = torch.cuda.get_device_name(device)
	if torch.cuda.is_available():
		print(f"GPU {torch.cuda.get_device_name()} available")
	else:
		print(f"No gpu acceleration available")

	data_config = config["data"]

	data_name = data_config["dataset"]
	h = data_config["img_size"]

	c, radius, object_size, aspect_ratio = get_params(data_name)

	w = int(h*aspect_ratio)
	near = 0.5*radius - object_size / 2
	far =   0.5*radius + object_size / 2
	
	network = config["network"]

	model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=config["encoding"], network_config=network).to(device)
	# model = FastNerf(config["encoding"], config["network"]).to(device)

	if not args.load_checkpoint:
		optim = config["optim"]
		seed = None if optim["random_seed"] else 0
		random.seed(seed)

		interval = config["output"].get("interval", 10)
		n_images = config["data"].get("n_images")
		intermediate_slices = config["output"].get("intermediate_slices", True)
		run_name = uuid.uuid4().hex[0:7] if config["output"]["path"] == "hash" else f"out_{n_images}"
		path_slices = f'{out_dir}/{run_name}'
		print(f"Output will be written to {path_slices}.")

		print(f"Loading training data...")
		training_dataset = BlenderDataset(data_name, data_config["transforms_file"], split="train", img_wh=(w,h), scale=object_size, n_chan=c, noise_level=data_config["noise_mean"], noise_sd=data_config["noise_sd"], n_train=data_config["n_images"])
		if optim["pixel_importance_sampling"]:
			pixel_weights = training_dataset.get_pixel_values()
			sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
			data_loader = DataLoader(training_dataset, batch_size=optim["batchsize"], sampler=sampler)
		else:
			data_loader = DataLoader(training_dataset, optim["batchsize"], shuffle=True)
		training_im = training_dataset[:w*h,6:]
		# my.write_img(, f"out/{run_name}-train-img-{args.noise:.0e}-{args.noise_sd:.0f}.png")

		model_optimizer = torch.optim.Adam(model.parameters(), lr=optim["learning_rate"])
		scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=optim["milestones"], gamma=optim["gamma"])

		print(f"Finished loading training data.  Training model on {device}...")
		now = time.monotonic()

		# TODO: implement speedups in here:
		# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
		training_loss_db = []
		loss_function=nn.MSELoss()
		with tqdm(range(optim["training_epochs"]), desc="Epochs") as t:
			for ep in t:
				for batch_num, batch in enumerate(data_loader):
					ray_origins = batch[:, :3].to(device)
					ray_directions = batch[:, 3:6].to(device)
					ground_truth_px_values = batch[:, 6].to(device)

					regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=near, hf=far, nb_bins=optim["samples_per_ray"])
					
					loss = loss_function(regenerated_px_values, ground_truth_px_values)
					model_optimizer.zero_grad()
					loss.backward()
					model_optimizer.step()
					training_loss_db.append(linear_to_db(loss.item()))
					if intermediate_slices and (batch_num % interval == 0):
						write_slices(model, device, ep, batch_num, config["output"], path_slices) # output slices during training
						interval *= 10 # intermediate slices only written during first epoch
				scheduler.step()
				torch.save(model.cpu(), 'nerf_model') # save after each epoch
				model.to(device)
				t.set_postfix(loss=f"{training_loss_db[-1]:.1f} dB per pixel")

		write_slices(model, device, ep, batch_num, config["output"], path_slices) 

		training_time = time.monotonic() - now
		timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")

		snapshot_path = f"checkpoints/{run_name}.pt"
		torch.save(model.state_dict(), snapshot_path)
		trained_model = model
		print(f"Training complete. Model weights saved as {snapshot_path}")
	else:
		run_name = args.load_checkpoint
		snapshot_path = f"checkpoints/{run_name}.pt"
		training_loss = None # no info
		trained_model = model
		trained_model.load_state_dict(torch.load(snapshot_path))
		trained_model.eval()

	has_gt = False
	if has_gt:
		phantom = np.load("jaw/jaw_phantom.npy")

	test_device = config["hardware"]["test"]
	trained_model.to(test_device)

	output = config["output"]
	
	if output["images"]:
		# no noise in test data
		testing_dataset = BlenderDataset(data_name, data_config["transforms_file"], split="test", img_wh=(w,h), scale=object_size, n_chan=c)
		for img_index in range(1):
			test_loss, imgs = test_model(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=test_device, nb_bins=output["samples_per_ray"], H=h, W=w)
			cpu_imgs = [img.data.reshape(h, w).clamp(0.0, 1.0).detach().cpu().numpy() for img in imgs]
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
			text = f"test loss: {test_loss}"

			my.write_imgs((cpu_imgs, training_loss_db, sigma, sigma_gt, px_vals), f'out/{run_name}_loss_{img_index}.png', text, show_training_img=False)


	if output["slices"]:
		# stich together slices into a video
		sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_x_%04d.png -vf \"hflip\" out/{run_name}_slices_x.mp4"
		os.system(sys_command)
		sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_y_%04d.png -vf \"hflip\" out/{run_name}_slices_y.mp4"
		os.system(sys_command)
		sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_z_%04d.png -vf \"hflip\" out/{run_name}_slices_z.mp4"
		os.system(sys_command)


	# is_voxel_grid = True if (data_name == 'jaw') else False 
	# MAX_BRIGHTNESS = 2.5 if (data_name == 'jaw') else 10

	