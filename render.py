import torch

def compute_accumulated_transmittance(alphas):
	accumulated_transmittance = torch.cumprod(alphas, 1)
	return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
					  accumulated_transmittance[:, :-1]), dim=-1)

def get_points_in_slice(z0,device):
	# initially z=0 slice
	xs = torch.linspace(-50, 50, steps=100, device=device)
	x,y = torch.meshgrid(xs, xs, indexing='xy')
	z = torch.tensor([z0], device=device).expand(10000).reshape(x.shape)
	r = torch.stack([x,y,z],dim=2).reshape(-1, 3)
	return r #[10000, 3]

def get_points_along_rays(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins, volumetric, regularise=False):
	device = ray_origins.device
	t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
	# Perturb sampling along each ray.
	mid = (t[:, :-1] + t[:, 1:]) / 2.
	lower = torch.cat((t[:, :1], mid), -1)
	upper = torch.cat((mid, t[:, -1:]), -1)
	u = torch.rand(t.shape, device=ray_origins.device)
	t = lower + (upper - lower) * u  # [batch_size, nb_bins]
	delta = t[:, 1:] - t[:, :-1] # [batch_size, nb_bins-1]
	x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
	x_f = x[:,:-1,:]
	return x_f.reshape(-1, 3), delta


def render_rays(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins, volumetric, regularise=False):
	x, delta = get_points_along_rays(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins, volumetric, regularise)
	sigma = nerf_model(x) # [batch_size*(nb_bins-1), 1]
	sigma = sigma.reshape(delta.shape)

	# accumulate
	if volumetric:
		alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
		weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
		c = weights.sum(dim=1)  # Pixel values
	else:
		alpha = (sigma * delta).unsqueeze(2)
		# we drop the last value here because the delta value is unbounded
		c = alpha[:,:-1,:].sum(dim=1).expand(-1, 3)/nb_bins

	return c
