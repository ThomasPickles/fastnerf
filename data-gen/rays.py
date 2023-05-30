import numpy as np
import torch
from convert_data import BlenderDataset
import helpers as my
from render import get_points_along_rays
from phantom import get_sigma_gt, world_to_local
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from cycler import cycler # to get colours to match in different plots


w,h = 275, 331
near = 163 - 60 # camera is on a circle 163 from origin at (x,y,0)
far = 163 + 60 # 50 is approx

phantom = np.load('jaw/jaw_phantom.npy')

testing_dataset = BlenderDataset('jaw', 'transforms_full_b', split="test", img_wh=(w,h), n_chan=3, noise_level=0, noise_sd=0)

img_index = 6 # np.random.randint(15)
view = testing_dataset[img_index]
gt = view[...,6:].squeeze(0)
gt = gt.reshape(h, w, 3)

NB_RAYS = 2
ray_ids = torch.tensor([29557, 50000]) # torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
# ray_ids = torch.tensor([0,5000,50000,51000,32000])

px_vals = my.get_px_values(ray_ids, w)

ray_origins = view[ray_ids,:3].squeeze(0)
ray_directions = view[ray_ids,3:6].squeeze(0)

points, delta = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=192)

first_ray = points[:192,:]
local_coords = world_to_local(first_ray)
# print(f"(x,y,z): {local_coords}")

sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
jitter = 0.005*np.random.rand(1,NB_RAYS)

print(np.concatenate((local_coords,sigma_gt[0,:].reshape(192,1)),axis=1))

# voxel at 
[[116,  84,  179]]


fig = plt.figure(tight_layout=True, figsize=(40., 20.))
gs = gridspec.GridSpec(1, 3)


# colors = np.random.rand(5)
colors = ['#1f77b4', '#ff7f0e'] #, '#2ca02c', '#d62728'] #, '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
custom_cycler = cycler(color=colors)

ax = fig.add_subplot(gs[0, 1])
ax.imshow(gt)
ax.scatter(px_vals[:,0]+0.5,px_vals[:,1]+0.5,c=colors,s=250)
ax.set_title("points")

# ray
ax = fig.add_subplot(gs[0, 2])
ax.set_prop_cycle(custom_cycler)
ADJUSTED_BRIGHTNESS = 1
ax.plot(ADJUSTED_BRIGHTNESS*sigma_gt.transpose()+jitter, '.-', lw=4)
ax.set_title("density along rays")

fig.suptitle("rays through phantom", fontsize=16)
plt.savefig("tmp", bbox_inches='tight')
plt.close()

print(img_index, ray_ids)