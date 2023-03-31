import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def write_imgs(imgs, path):
	fig = plt.figure(figsize=(30., 10.))
	grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 3),
                 axes_pad=0.1,  # pad between axes in inch.
                 )

	for ax, im in zip(grid, imgs):
		ax.imshow(im.clip(0,1))

	plt.axis('off')
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