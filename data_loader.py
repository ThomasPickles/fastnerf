import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import json
import numpy as np
import os
from torchvision import transforms as T
from torchvision import io
import random

from ray_utils import *
from img_helpers import NerfImage
from chart_writer import write_img
import img_transforms 

class ImagesDataset(Dataset):
    def __init__(self, data_name, filename, split, img_wh, scale, n_chan, noise_level=0, n_train=0):
        self.data_name = data_name
        self.root_dir = f'data/{data_name}'
        self.filename = filename
        self.split = split
        self.n_chan = n_chan
        self.scale = scale
        self.noise_level = noise_level
        self.img_wh = img_wh
        self.n_train = n_train
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"{self.filename}_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        # This is not really a focal length, it's a rescaling to ensure that
        # the object of interest is centred in the [0,1]^3 cube
        detector_width =  self.img_wh[0] 
        # Checking the z-positions of the rays, they are much too high
        self.focal = 0.5*detector_width/np.tan(0.5*self.meta['camera_angle_x'])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions, self.pix_x, self.pix_y = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (h, w, 3)

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
            self.data = torch.cat((self.all_rays[:,:6],self.all_rgbs[:,:]),1)

            print(f"Training data points: {self.data.shape[0]}")

    def process_rays(self, frame):
        pose = np.array(frame['transform_matrix'])[:3, :4]
        self.poses += [pose]
        c2w = torch.FloatTensor(pose)
        rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
        # rescale scene so that object fits
        # in a [0,1] box   
        # no longer have normalised rays
        rays_o = 0.5 * rays_o / self.scale + 0.5 
        rays_d = rays_d / self.scale
        return rays_o, rays_d


    def process_image(self, frame):
        image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
        self.image_paths += [image_path]

        if self.data_name == 'walnut':
            transform = img_transforms.walnut
        if self.data_name     == 'jaw':
            transform = img_transforms.jaw
        
        img = NerfImage(image_path, transform, self.noise_level, (self.img_wh[1], self.img_wh[0]))

        px = []
        x_vals = torch.flatten(self.pix_x).numpy() # [h*w]
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
