import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from torchvision import io
import imageio # can read 16-bit tif files, but PIL can't yet
import random

from ray_utils import *
from img_helpers import NerfImage
from helpers import write_img

class BlenderDataset(Dataset):
    def __init__(self, root_dir, filename, split, img_wh, n_chan, noise_level=0, noise_sd=0, n_train=0):
        self.root_dir = root_dir
        self.filename = filename
        self.split = split
        self.n_chan = n_chan
        self.noise_level = noise_level
        self.noise_sd = noise_sd
        self.img_wh = img_wh
        self.n_train = n_train
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"{self.filename}_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w_orig = self.meta['w']
        h_orig = self.meta['h']
        w, h = self.img_wh

        # NOTE: w_orig just drops out of this
        self.focal = 0.5*w_orig/np.tan(0.5*self.meta['camera_angle_x'])
        self.focal *= w/w_orig # modify focal length to match size self.img_wh
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions, self.pix_x, self.pix_y = get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        
        if (self.split == 'train'):
            
            frames = self.meta['frames']
            for frame in random.sample(frames, self.n_train):

                rays_o, rays_d = self.process_rays(frame)
                img = self.process_image(frame)
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d],1)] # (h*w, 6)
            
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            assert torch.max(self.all_rgbs) <= 1.0 # should all be between 0 and 1
            # NOTE: we could put the near and far bounds in here too, if we want
            self.data = torch.cat((self.all_rays[:,:6],self.all_rgbs[:,:]),1)
            print(f"Training data shape: {self.data.shape}")

    def process_rays(self, frame):
        pose = np.array(frame['transform_matrix'])[:3, :4]
        # print(f"pose (BEFORE): {pose}")
        # pose[:,3] = pose[:,3] / self.scale # RESCALE POSITION
        # print(f"pose (AFTER): {pose}")
        self.poses += [pose]
        c2w = torch.FloatTensor(pose)
        rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
        # rescale origin
        return rays_o, rays_d


    def process_image(self, frame):
        # TODO: tif or png!
        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.tif")
        self.image_paths += [image_path]
        img_transform = lambda img: -np.log(img)
        img = NerfImage(image_path, img_transform)
        # if self.n_chan == 3: # no alpha
        #     # just taking red channel here!
        #     img = img[:,1]
        # elif self.n_chan == 1:
        #     #TODO: walnut imgs needs to be shifted 0.22% to the left
        #     img = img.squeeze() 
        # assert img.shape == (h, w)

        # noise = Image.effect_noise(self.img_wh, self.noise_sd)
        # noise = self.transform(noise).squeeze()
        # img = (img + self.noise_level*(noise - 0.5)).clamp(0,1) # noise centred at 0.5

        # TODO: add bilinear interpolation into training data
        # jitter = torch.rand_like(self.directions[1:-1,1:-1,:]) - 0.5
        # jitter[...,2] = 0 # no in-plane jitter
        # jx = jitter[...,0]
        # jy = jitter[...,1]
        # assert jx.shape == jy.shape
        # interpolated_img = torch.zeros_like(img)
        # # bilinear interpolation, ignore cells around the boundary
        # interpolated_img[1:-1,1:-1] = (1-jx)*(1-jy)*img[:-1,:-1] + (1-jx)*jy*img[:-1,1:] + jx*(1-jy)*img[1:,:-1] + jx*jy*img[1:,1:]
        # img = interpolated_img
        # rays_o, rays_d = get_rays(self.directions + jitter / self.focal, c2w) # both (h*w, 3)
        
        px = []
        x_vals = torch.flatten(self.pix_x).numpy()
        y_vals = torch.flatten(self.pix_y).numpy()
        for x_val, y_val in zip(x_vals, y_vals): 
            px += [img.get_pixel_normalised(x_val, y_val)]

        img = torch.tensor(px)

        img = img.view(1, -1).permute(1, 0) # (h*w, 1) RGB
        assert img.shape == (self.img_wh[1]* self.img_wh[0], 1)

        return img


    def define_transforms(self):
        self.transform = T.ToTensor()

    def get_pixel_values(self):
        return self.data[:,6]

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test':
            return 1

    def __getitem__(self, idx):
        if (self.split == 'train'):
            return self.data[idx]
        if ((self.split == 'test') or (self.split == 'video')):
            frame = self.meta['frames'][idx]
            rays_o, rays_d = self.process_rays(frame)
            img = self.process_image(frame)
            rays = torch.cat([rays_o, rays_d, img],1)
            return rays
