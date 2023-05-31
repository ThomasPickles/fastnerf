import torch

# see 8b2f, slice 50
def get_samples_around_point(points, box_size, nb_samples):
	# input is essentially a regular 2d grid 
	# [nb_points, 3]
	# create a cube centred on the point, and sample uniformly within it
	nb_points = points.shape[0]
	samples = points.unsqueeze(0)
	jitter = box_size*(torch.rand((nb_samples, nb_points, 3), device=points.device) - 0.5)
	# output
	# [nb_samples, nb_points, 3]
	return samples + jitter

if __name__ == '__main__':

	points = torch.tensor(
		[[1,2,3],
		[8,9,10]]
		)
	print(points.shape)
	nb_samples = 2
	point_extent = 1.

	output = get_samples_around_point(points, point_extent, nb_samples)
	print(output)

	assert output.shape == (4,3)
	assert torch.max(output[0,:] - points[0,:]) <= 0.5

