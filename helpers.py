import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec

def write_imgs(data, path, title=None):
	fig = plt.figure(figsize=(40., 10.))
	# fig = plt.figure(tight_layout=True)
	gs = gridspec.GridSpec(1, 5)
	# grid = ImageGrid(fig, 111,
	# 			 nrows_ncols=(1, 4),
	# 			 axes_pad=0.1,  # pad between axes in inch.
	# 			 aspect = True
	# 			 )
	imgs, curve = data

	# images
	for i, sub_tit in enumerate(["out","ref","diff"]):
		ax = fig.add_subplot(gs[0, i])
		ax.imshow(imgs[i].clip(0,1))
		ax.set_title(sub_tit)

	# curve
	ax = fig.add_subplot(gs[0, 3:5])
	ax.plot(curve)
	ax.set_ylabel('dB')
	ax.invert_yaxis()

	fig.suptitle(title, fontsize=16)
	plt.savefig(path, bbox_inches='tight')
	plt.close()

def write_img(img, path):
	plt.figure(figsize=(10., 10.))
	plt.imshow(img)
	plt.axis('off')
	plt.savefig(path, bbox_inches='tight')
	print(f"Image written to {path}")
	plt.close()

def linear_to_db(x):
	return -10.*np.log(x)/np.log(10.)