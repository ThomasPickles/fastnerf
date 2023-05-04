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

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("-e", "--epochs", type=int, default=2, help="The number of training epochs to run")
	parser.add_argument("-l", "--layers", type=int, default=8, help="Number of hidden layers")
	parser.add_argument("-d", "--encoding_dim", type=int, default=10, help="Number of hidden layers")
	parser.add_argument("-n", "--neurons", type=int, default=384, help="Neurons per layer")
	parser.add_argument("-s", "--samples", type=int, default=192, help="Number of samples per ray")
	parser.add_argument("--n_train", default=200, type=int, help="Network learning rate")
	parser.add_argument("--lr", default=1e-4, type=float, help="Network learning rate")
	parser.add_argument("--noise", default=0, type=float, help="Gaussian noise level to apply to training images")
	parser.add_argument("--noise_sd", default=128, type=float, help="Gaussian noise level to apply to training images")
	parser.add_argument("--loss", default='L2', choices=['L2','Huber','L1','Exp'], help="Loss function")
	parser.add_argument("--height", "--px", type=int, default=150, help="Compressed image height")
	parser.add_argument("--batchsize", type=int, default=1024, help="Number of training steps before update params")
	parser.add_argument("--dataset", default='walnut', choices=['jaw','walnut'])
	parser.add_argument("--phantom_path", default='jaw/jaw_phantom.npy')
	parser.add_argument("--device", default='cuda', choices=['cuda','cpu'])
	parser.add_argument("--test_device", default='cpu', choices=['cuda','cpu'])
	parser.add_argument("--file", default='transforms')
	parser.add_argument("--importance_sampling", action='store_true', help="Privileges bright pixels for training")
	parser.add_argument("--video", action='store_true', help="Outputs video")
	parser.add_argument("--slice", action='store_true', help="Outputs slice")
	parser.add_argument("--load_checkpoint", default='', help="Load uuid and bypass training")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	device = args.device
	device_name = torch.cuda.get_device_name(device)

	h = args.height
	if h < 100:
		print(f"Warning: low res images can lead to jaggies in training")

	c, w, (near,far) = get_params(args.dataset,h)

	random.seed(0)

	if not args.load_checkpoint:
		checkpoint = uuid.uuid4().hex[0:10]
		
		training_dataset = BlenderDataset(args.dataset, args.file, split="train", img_wh=(w,h), n_chan=c, noise_level=args.noise, noise_sd=args.noise_sd, n_train=args.n_train)
		if args.importance_sampling:
			pixel_weights = training_dataset.get_pixel_values()
			sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
			data_loader = DataLoader(training_dataset, batch_size=args.batchsize, sampler=sampler)
		else:
			data_loader = DataLoader(training_dataset, args.batchsize, shuffle=True)
		training_im = training_dataset[:w*h,6:]
		# my.write_img(, f"out/{checkpoint}-train-img-{args.noise:.0e}-{args.noise_sd:.0f}.png")

		model = FastNerf(args.encoding_dim, args.layers, args.neurons).to(device)

		model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
		# i = 0
		# data = iter(data_loader)
		# all_vals = []
		# while i < 100:
		# 	this_batch = next(data)
		# 	px_values = this_batch[:,6]
		# 	all_vals.extend(px_values)
		# 	# print(f"batch {i}, with shape {this_batch.shape} has px_vals {px_values}")
		# 	i = i+1
		# import matplotlib.pyplot as plt
		# plt.hist(np.array(all_vals), bins=10)
		# plt.show()
		# exit()
		
		if args.loss == 'L2':
			loss_function = nn.MSELoss()
		elif args.loss == 'Huber':
			loss_function = nn.HuberLoss()
		elif args.loss == 'L1':
			loss_function = nn.L1Loss()
		elif args.loss == 'Exp':
			loss_function = ExpLoss
		else:
			print('Loss not implemented')
			exit()

		now = time.monotonic()
		training_loss = train(model, model_optimizer, scheduler, data_loader, nb_epochs=args.epochs, device=device, hn=near, hf=far, nb_bins=args.samples, loss_function=loss_function)
		training_time = time.monotonic() - now
		timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")

		params_dict = {
				"timestamp": timestamp,
				"model": model.toDict(),
				"final_training_loss_db": format(training_loss[-1], ".2f"),
				"epochs": args.epochs,
				"img_size": h,
				"rendering": "astra",
				"samples": args.samples,
				"lr": args.lr,
				"batchsize": args.batchsize,
				"loss": args.loss,
				"run_time": format(training_time, ".2f"),
				"device": device_name,
				"checkpoint": checkpoint, 
				"training_loss": training_loss,
			}
		with open("run.log", 'a') as logfile:
			json.dump(params_dict, logfile, default=lambda o: o.__dict__)
			logfile.write('\n')
		with open(f"checkpoints/{checkpoint}.json", 'w') as jsonfile:
			json.dump(params_dict, jsonfile, default=lambda o: o.__dict__)
		
		snapshot_path = f"checkpoints/{checkpoint}.pt"
		torch.save(model.state_dict(), snapshot_path)
		print(f"Training complete. Model weights saved as {snapshot_path}")
	else:
		checkpoint = args.load_checkpoint
		snapshot_path = f"checkpoints/{checkpoint}.pt"

	with open(f"checkpoints/{checkpoint}.json", 'r') as jsonfile:
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
	
	if lr == None:
		print("No learning rate.  Skipping file...")
		exit()

	trained_model = FastNerf(embed_dim, layers, neurons)
	trained_model.load_state_dict(torch.load(snapshot_path))
	trained_model.eval()
	trained_model.to(args.test_device)

	is_voxel_grid = True if (args.dataset == 'jaw') else False 
	MAX_BRIGHTNESS = 2.5 if (args.dataset == 'jaw') else 10
	if args.slice:
		for idx in range(100):
			z = int(3.3*idx) if is_voxel_grid else idx - 50
			resolution = (100,100)
			img = render_slice(model=trained_model, z=z, device=args.test_device, resolution=resolution, voxel_grid=is_voxel_grid)
			img = img.data.cpu().numpy().reshape(resolution[0], resolution[1], 3)/MAX_BRIGHTNESS
			my.write_img(img, f'tmp/slice_{checkpoint}_{idx:03}.png', verbose=False)
		sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/slice_{checkpoint}_%03d.png out/{checkpoint}_slices_{epochs}_{img_size}_{layers}_{neurons}.mp4"
		os.system(sys_command)

	phantom = np.load(args.phantom_path)

	# no noise in test data
	testing_dataset = BlenderDataset(args.dataset, 'transforms', split="test", img_wh=(w,h), n_chan=c)
	
	for img_index in range(3):
		test_loss, imgs = batch_test(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=args.test_device, nb_bins=args.samples, H=h, W=w)
		cpu_imgs = [img.data.cpu().numpy().reshape(h, w, 3) for img in imgs]
		train_img = training_im.reshape(h, w, 3)
		cpu_imgs.append(train_img)
		view = testing_dataset[img_index]
		
		NB_RAYS = 5
		ray_ids = torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
		px_vals = my.get_px_values(ray_ids, w) 
	
		ray_origins = view[ray_ids,:3].squeeze(0)
		ray_directions = view[ray_ids,3:6].squeeze(0)

		points, delta = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=args.samples)
	
		# since x are calculated randomly, we need to pass in the same values
		sigma = get_ray_sigma(trained_model, points, device=args.test_device)
		
		has_gt = False # TODO: fix ground truth True if (args.dataset == 'jaw') else False 
		if has_gt:
			sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
			sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
			jitter = 0.005*np.random.rand(NB_RAYS,1)
			sigma_gt = sigma_gt + jitter # add some jitter to distinguish values
		else:
			sigma_gt = None

		sigma = sigma.data.cpu().numpy().reshape(NB_RAYS,-1)
		text = f"test_loss: {test_loss:.1f}dB, training_loss: {float(final_training_loss_db):.1f}dB, lr: {lr:.2E}, loss function: {loss}, training noise level (sd): {args.noise} ({args.noise*(args.noise_sd/256)})\nepochs: {epochs}, layers: {layers}, neurons: {neurons}, embed_dim: {embed_dim}, training time (h): {training_time/3600:.2f}\nnumber of training images: {args.n_train}, img_size: {img_size}, samples per ray: {samples}, pixel importance sampling: {args.importance_sampling}\n"
		my.write_imgs((cpu_imgs,curve, sigma, sigma_gt, px_vals), f'out/{checkpoint}_loss_{img_index}.png', text, show_training_img=True)

	if args.video:
		video_dataset = BlenderDataset(args.dataset, 'transforms', split="video", img_wh=(w,h), n_chan=c)

		for img_index in range(720):
			_, imgs = batch_test(model=trained_model, dataset=video_dataset, img_index=img_index, hn=near, hf=far, device=args.test_device, nb_bins=args.samples, H=h, W=w)
			img = imgs[0].data.cpu().numpy().reshape(h, w, 3)
			my.write_img(img, f'tmp/rot_{checkpoint}_{img_index:03}.png', verbose=False)
		sys_command = f"ffmpeg -hide_banner -loglevel error -i tmp/rot_{checkpoint}_%03d.png out/{checkpoint}_rotate_{epochs}_{img_size}_{layers}_{neurons}.mp4"
		os.system(sys_command)
