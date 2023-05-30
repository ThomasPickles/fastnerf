import numpy as np
import os
import torch
from convert_data import BlenderDataset
import helpers as my
from render import get_points_along_rays
from phantom import get_sigma_gt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from cycler import cycler # to get colours to match in different plots

# phantom is z,y,x
phantom = np.load('jaw/jaw_phantom.npy')
h,w,_ = phantom.shape

assert phantom.shape == (h,w,w)
# # mark the zero planes with white
# phantom[0:5,:,:] = 1
# phantom[:,0:5,:] = 1
# phantom[:,:,0:5] = 1

# power_norm=colors.PowerNorm(gamma=0.5, vmin=0, vmax=1)

# # ax1 slices - Top to bottom
# fig_1 = plt.figure(tight_layout=True, figsize=(40., 20.))
# fig_1.patch.set_facecolor('black')
# gs_1 = gridspec.GridSpec(5, 12)
# for i in range(5): #in range(h):
# 	for j in range(12):
# 		ax = fig_1.add_subplot(gs_1[i, j])
# 		ax.axis('off')
# 		z = 20+ 5*(12*i + j)	
# 		jaw_slice = phantom[z,:,:]
# 		ax.text(10, 50, z, fontsize=36, color='blue')
# 		ax.imshow(jaw_slice,cmap='Greys_r', norm=power_norm)
# plt.savefig("slices_axis_1")

for z in range(331):
	jaw_slice = phantom[z,:,:]
	img = np.broadcast_to(np.expand_dims(jaw_slice, 2),(275,275,3))
	my.write_img(10*img, f'tmp/slice_gt_{z:03}.png', verbose=False)
sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/slice_gt_%03d.png out/gt_slices.mp4"
os.system(sys_command)

# # ax2 slices - Front to back
# fig_2 = plt.figure(tight_layout=True, figsize=(40., 20.))
# fig_2.patch.set_facecolor('black')
# gs_2 = gridspec.GridSpec(6, 13)
# for i in range(6): #in range(h):
# 	for j in range(13):
# 		y = 24+3*(13*i + j)	
# 		jaw_slice = phantom[:,y,:]
# 		ax = fig_2.add_subplot(gs_2[i, j])
# 		ax.text(10, 50, y, fontsize=36, color='green')
# 		ax.axis('off')
# 		ax.imshow(jaw_slice,cmap='Greys_r', norm=power_norm)
# plt.savefig("slices_axis_2")

# # x slices - Right to left
# fig_3 = plt.figure(tight_layout=True, figsize=(40., 20.))
# fig_3.patch.set_facecolor('black')
# gs_3= gridspec.GridSpec(5, 13)
# for i in range(5): #in range(h):
# 	for j in range(13):
# 		x = 50+ 3*(13*i + j)	
# 		jaw_slice = phantom[:,:,x]
# 		ax = fig_3.add_subplot(gs_3[i, j])
# 		ax.text(10, 50, x, fontsize=36, color='red')
# 		ax.axis('off')
# 		ax.imshow(jaw_slice,cmap='Greys_r', norm=power_norm)
# plt.savefig("slices_axis_3")
