#generate data for nerf
import numpy as np 
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10),
                        layout="constrained")
# add an artist, in this case a nice label in the middle...

for col, n in enumerate([10**1, 10**2, 10**3, 10**4]):
    ax = axs[col] 
    img = np.asarray(Image.open(f'../fastnerf/config/test_config/it_{n}.png'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img[100:-100,100:-100])
    ax.set_title(n, fontsize=20)
ax = axs[4] 
img = np.asarray(Image.open(f'../fastnerf/tmp/slice_gt_092.png'))
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(np.flip(img,1), origin='lower')
ax.set_title('GT', fontsize=20)

plt.show()
# fig.savefig(f'out/reconstruction.png')
# plt.close()