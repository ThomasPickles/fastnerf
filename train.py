import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from convert_data import BlenderDataset
from nerf import FastNerf
from render import render_rays
from datasets import get_params

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e3),
		  nb_bins=192):
	training_loss = []
	print(f"Training on {torch.cuda.get_device_name()}")
	for _ in (range(nb_epochs)):
		for ep, batch in enumerate(tqdm(data_loader)):
			ray_origins = batch[:, :3].to(device)
			ray_directions = batch[:, 3:6].to(device)
			ground_truth_px_values = batch[:, 6:].to(device)

			regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
			loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			training_loss.append(loss.item())
		scheduler.step()
		torch.save(nerf_model.cpu(), 'nerf_model') # save after each epoch
		nerf_model.to(device)
	return training_loss

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument("--epochs", type=int, default=8, help="The number of training epochs to run")
	parser.add_argument("--px", type=int, default=40, help="The image height to train on")
	parser.add_argument("--dataset", default='lego', choices=['lego','jaw'])
	parser.add_argument("--device", default='cuda', choices=['cuda','cpu'])
	parser.add_argument("--snapshot", default='snapshot.pt', help="The name of the snapshot")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	device = args.device

	h = args.px
	dataset = args.dataset
	c, w, (near,far) = get_params(dataset,h)

	# training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
	training_dataset = BlenderDataset(dataset, img_wh=(w,h), n_chan=c)

	model = FastNerf().to(device)

	model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)


	data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
	train(model, model_optimizer, scheduler, data_loader, nb_epochs=args.epochs, device=device, hn=near, hf=far)
	torch.save(model.state_dict(), args.snapshot)
