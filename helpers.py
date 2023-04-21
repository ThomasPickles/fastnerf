import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from cycler import cycler # to get colours to match in different plots

def get_px_values(ray_ids, W):
	cols = np.remainder(ray_ids, W)
	rows = np.floor(ray_ids / W)
	px_vals = np.stack((cols,rows), axis=1)
	return px_vals

def write_imgs(data, path, title=None):
	fig = plt.figure(tight_layout=True, figsize=(40., 20.))
	gs = gridspec.GridSpec(2, 5)
	imgs, curve, rays, rays_gt, px_vals = data

	# colors = np.random.rand(5)
	colors = ['b','g','r','c','m']
	custom_cycler = cycler(color=colors)

	# images
	for i, sub_tit in enumerate(["out","test","diff","example training"]):
		ax = fig.add_subplot(gs[0, i])
		ax.imshow(imgs[i].clip(0,1))
		if i < 2:
			ax.scatter(px_vals[:,0],px_vals[:,1],c=colors,s=250)
		ax.set_title(sub_tit)

	# curve
	ax = fig.add_subplot(gs[1, :])
	ax.plot(curve)
	ax.set_ylabel('dB')
	ax.invert_yaxis()

	# ray
	ax = fig.add_subplot(gs[0, 4])
	ax.set_prop_cycle(custom_cycler)
	ax.plot(rays.transpose(), '-')
	ADJUSTED_BRIGHTNESS = 10
	ax.plot(ADJUSTED_BRIGHTNESS*rays_gt.transpose(), '--')
	ax.set_title("density along rays")

	fig.suptitle(title, fontsize=16)
	plt.savefig(path, bbox_inches='tight')
	print(f"Image written to {path}")
	plt.close()

def write_img(img, path, verbose=True):
	plt.figure(figsize=(10., 10.))
	plt.imshow(img)
	plt.axis('off')
	plt.savefig(path, bbox_inches='tight')
	if verbose:
		print(f"Image written to {path}")
	plt.close()

def linear_to_db(x):
	return -10.*np.log(x)/np.log(10.)