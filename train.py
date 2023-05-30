import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from render import render_rays
from helpers import linear_to_db

def train(nerf_model, optimizer, scheduler, data_loader, device, hn, hf, nb_epochs, nb_bins, loss_function):
	training_loss_db = []
	with tqdm(range(nb_epochs), desc="Epochs") as t:
		for _ in t:
			for ep, batch in enumerate(data_loader):
				ray_origins = batch[:, :3].to(device)
				ray_directions = batch[:, 3:6].to(device)
				ground_truth_px_values = batch[:, 6].to(device)

				regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
				
				loss = loss_function(regenerated_px_values, ground_truth_px_values)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				training_loss_db.append(linear_to_db(loss.item()))
			scheduler.step()
			torch.save(nerf_model.cpu(), 'nerf_model') # save after each epoch
			nerf_model.to(device)
			t.set_postfix(loss=f"{training_loss_db[-1]:.1f} dB per pixel")
	return nerf_model, training_loss_db
