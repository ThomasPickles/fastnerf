import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import time, json, uuid

from convert_data import BlenderDataset
from nerf import FastNerf
from render import render_rays
from datasets import get_params
from helpers import linear_to_db

# TODO: cross-validation of model params

def ExpLoss(output, target):
	return torch.pow(torch.exp(output) - torch.exp(target), 2.).mean()

def train(nerf_model, optimizer, scheduler, data_loader, device, hn, hf, nb_epochs, nb_bins, loss_function):
	training_loss_db = []
	with tqdm(range(nb_epochs), desc="Epochs") as t:
		for _ in t:
			for ep, batch in enumerate(data_loader):
				ray_origins = batch[:, :3].to(device)
				ray_directions = batch[:, 3:6].to(device)
				ground_truth_px_values = batch[:, 6:].to(device)

				regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, volumetric=False)
				
				loss = loss_function(regenerated_px_values, ground_truth_px_values)
				# perhaps a better loss would be relative error?
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				training_loss_db.append(linear_to_db(loss.item()))
			scheduler.step()
			torch.save(nerf_model.cpu(), 'nerf_model') # save after each epoch
			nerf_model.to(device)
			t.set_postfix(loss=f"{training_loss_db[-1]:.1f} dB per pixel")
	return training_loss_db


def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("-e", "--epochs", type=int, default=2, help="The number of training epochs to run")
	parser.add_argument("-l", "--layers", type=int, default=8, help="Number of hidden layers")
	parser.add_argument("-d", "--encoding_dim", type=int, default=10, help="Number of hidden layers")
	parser.add_argument("-n", "--neurons", type=int, default=132, help="Neurons per layer")
	parser.add_argument("-s", "--samples", type=int, default=192, help="Number of samples per ray")
	parser.add_argument("--lr", default=1e-4, type=float, help="Network learning rate")
	parser.add_argument("--loss", default='L2', choices=['L2','Huber','L1','Exp'], help="Loss function")
	parser.add_argument("--height", "--px", type=int, default=150, help="Compressed image height")
	parser.add_argument("--batchsize", type=int, default=1024, help="Number of training steps before update params")
	parser.add_argument("--dataset", default='jaw', choices=['lego','jaw'])
	parser.add_argument("--device", default='cuda', choices=['cuda','cpu'])
	parser.add_argument("--file", default='transforms_full')

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
	print(f"Trained model weights saved as checkpoints/{checkpoint}.pt")
	torch.save(model.state_dict(), f"checkpoints/{checkpoint}.pt")
