import torch

def compute_accumulated_transmittance(alphas):
	accumulated_transmittance = torch.cumprod(alphas, 1)
	return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
					  accumulated_transmittance[:, :-1]), dim=-1)

def get_voxels_in_slice(z0, device, resolution):
	# initially z=0 slice
	# actually z slices are square!
	# JAW DIMENSIONS
	xs = torch.linspace(0, 275, steps=resolution[0], device=device)
	ys = torch.linspace(0, 275, steps=resolution[1], device=device)
	n_pixels = resolution[0] * resolution[1]
	x,y = torch.meshgrid(xs, ys, indexing='xy')
	z = torch.tensor([z0], device=device).expand(n_pixels).reshape(x.shape)
	voxs = torch.stack([z,y,x],dim=2).reshape(-1, 3)
	return voxs 

def get_points_in_slice(dim, device, resolution):
	half_dx = 1./resolution[0]
	half_dy = 1./resolution[1]
	d1s = torch.linspace(0.+half_dx, 1.-half_dx, steps=resolution[0], device=device)
	d2s = torch.linspace(0.+half_dy, 1.-half_dy, steps=resolution[1], device=device)
	n_pixels = resolution[0] * resolution[1]
	d1, d2 = torch.meshgrid(d1s, d2s, indexing='xy')
	d0 = torch.zeros_like(d1) # ([z0], device=device).expand(n_pixels).reshape(x.shape)
	d0 = d0 + 0.5 # object centred at 0.5
	if dim == 0: # x=0
		points = torch.stack([d0,d1,d2], dim=2).reshape(-1, 3)
	if dim == 1: # y=0
		points = torch.stack([d1,d0,d2], dim=2).reshape(-1, 3)
	if dim == 2: # z=0
		points = torch.stack([d1,d2,d0], dim=2).reshape(-1, 3)
	return points 


def get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins):
	device = ray_origins.device
	t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
	# Perturb sampling along each ray.
	mid = (t[:, :-1] + t[:, 1:]) / 2.
	lower = torch.cat((t[:, :1], mid), -1)
	upper = torch.cat((mid, t[:, -1:]), -1)
	u = torch.rand(t.shape, device=device)
	t = lower + (upper - lower) * u  # [batch_size, nb_bins]
	delta = t[:, 1:] - t[:, :-1] # [batch_size, nb_bins-1]
	x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
	print(f"ray near/far is {x[1,0,:]}/{x[1,-1,:]	}")
	return x.reshape(-1, 3), delta


def render_rays(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins):
	x, delta = get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins)
	sigma = nerf_model(x) # [batch_size*(nb_bins-1), 1]
	sigma = sigma.reshape(-1, nb_bins)

	# interpolate: nb_bins values, nb_bins - 1 intervals
	# --- | ------- | ----- | --- | --------
	# --- | ------- | ----- | --- | --------
	sigma_mid = (sigma[:, 1:] + sigma[:, :-1])/2
	alpha = (sigma_mid * delta).unsqueeze(2)
	# single channel
	c = alpha.sum(dim=1).squeeze()/nb_bins

	return c
