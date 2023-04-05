import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import time, json, uuid
import os

from convert_data import BlenderDataset
from nerf import FastNerf
from datasets import get_params
from helpers import linear_to_db
from test import batch_test, get_ray_alpha, render_slice
from train import train
from helpers import *

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("-e", "--epochs", type=int, default=2, help="The number of training epochs to run")
	parser.add_argument("-l", "--layers", type=int, default=8, help="Number of hidden layers")
	parser.add_argument("-d", "--encoding_dim", type=int, default=10, help="Number of hidden layers")
	parser.add_argument("-n", "--neurons", type=int, default=384, help="Neurons per layer")
	parser.add_argument("-s", "--samples", type=int, default=192, help="Number of samples per ray")
	parser.add_argument("--lr", default=1e-4, type=float, help="Network learning rate")
	parser.add_argument("--loss", default='L2', choices=['L2','Huber','L1','Exp'], help="Loss function")
	parser.add_argument("--height", "--px", type=int, default=150, help="Compressed image height")
	parser.add_argument("--batchsize", type=int, default=1024, help="Number of training steps before update params")
	parser.add_argument("--dataset", default='jaw', choices=['lego','jaw'])
	parser.add_argument("--device", default='cuda', choices=['cuda','cpu'])
	parser.add_argument("--test_device", default='cpu', choices=['cuda','cpu'])
	parser.add_argument("--file", default='transforms_full_b')
	parser.add_argument("--video", action='store_true', help="Outputs video")
	parser.add_argument("--slice", action='store_true', help="Outputs slice")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	device = args.device
	device_name = torch.cuda.get_device_name(device)

	h = args.height
	if h < 100:
		print(f"Warning: low res images can lead to jaggies in training")

	c, w, (near,far) = get_params(args.dataset,h)

	training_dataset = BlenderDataset(args.dataset, args.file, split="train", img_wh=(w,h), n_chan=c)

	model = FastNerf(args.encoding_dim, args.layers, args.neurons).to(device)

	model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
	data_loader = DataLoader(training_dataset, batch_size=args.batchsize, shuffle=True)
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
		exit

	now = time.monotonic()
	training_loss = train(model, model_optimizer, scheduler, data_loader, nb_epochs=args.epochs, device=device, hn=near, hf=far, nb_bins=args.samples, loss_function=loss_function)
	training_time = time.monotonic() - now
	timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")
	checkpoint = uuid.uuid4().hex[0:10]
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
	
	print(f"Run {checkpoint} complete with results {json.dumps(params_dict)} and written to run.log")
	snapshot_path = f"checkpoints/{checkpoint}.pt"
	torch.save(model.state_dict(), snapshot_path)
	print(f"Trained model weights saved as checkpoints/{checkpoint}.pt")

	# TODO: do we need intermediate read?
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

	if args.slice:
		for z in [0,10,20,30,40,50,60,70]:
			img = render_slice(model=trained_model, z=z, device=args.test_device)
			img = img.data.cpu().numpy().reshape(100, 100, 3)
			write_img(img, f'slices/img_{checkpoint}_{z:.0f}.png')

	testing_dataset = BlenderDataset(args.dataset, 'transforms_full_b', split="test", img_wh=(w,h), n_chan=c)

	for img_index in range(4):
		test_loss, imgs = batch_test(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=args.test_device, nb_bins=samples, H=h, W=w)
		weights = get_ray_alpha(trained_model, testing_dataset, img_index, hn=near, hf=far, device=args.test_device, nb_bins=samples, H=h, W=w)
		cpu_imgs = [img.data.cpu().numpy().reshape(h, w, 3) for img in imgs]
		text = f"test_loss: {test_loss:.1f}dB, training_loss: {final_training_loss_db}dB\nlr: {lr}, loss function: {loss}, epochs: {epochs}\nlayers: {layers}, neurons: {neurons}, embed_dim: {embed_dim}, img_size: {img_size},\nrendering: {rendering}, samples: {samples}"
		write_imgs((cpu_imgs,curve, weights.data.cpu()), f'novel_views/img_{checkpoint}_{img_index}.png', text)

	if args.video:
		video_dataset = BlenderDataset(args.dataset, 'transforms_full_b', split="video", img_wh=(w,h), n_chan=c)

		for img_index in range(720):
			_, imgs = batch_test(model=trained_model, dataset=video_dataset, img_index=img_index, hn=near, hf=far, device=args.test_device, nb_bins=samples, H=h, W=w)
			img = imgs[0].data.cpu().numpy().reshape(h, w, 3)
			write_img(img, f'video/img_{checkpoint}_{img_index:03}.png')
		sys_command = f"ffmpeg -i video/img_{checkpoint}_%03d.png video/vid_{epochs}_{img_size}_{layers}.mp4"
		os.system(sys_command)




