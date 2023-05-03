import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from torchvision import io

from ray_utils import *
from helpers import write_img

class BlenderDataset(Dataset):
    def __init__(self, root_dir, filename, split, img_wh, n_chan, noise_level=0, noise_sd=132):
        self.root_dir = root_dir
        self.filename = filename
        self.split = split
        self.n_chan = n_chan
        self.noise_level = noise_level
        self.noise_sd = noise_sd
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"{self.filename}_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w_orig = self.meta['w']
        h_orig = self.meta['h']
        w, h = self.img_wh

        self.focal = 0.5*w_orig/np.tan(0.5*self.meta['camera_angle_x'])
        self.focal *= w/w_orig # modify focal length to match size self.img_wh
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        
        if (self.split == 'train'):
            # since we add random noise to the images, we need
            # to save them all to self.data, since otherwise
            # if we added noise a second time (for multiple epochs
            # for example) it would be a different training image,
            # and although it would probably converge better,
            # it wouldn't be a fair test.
            for frame in self.meta['frames']:
                rays_o, rays_d, img = self.process_frame(frame)
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d],1)] # (h*w, 6)
                

            
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            assert torch.max(self.all_rgbs) <= 1.0 # should all be between 0 and 1
            # NOTE: we could put the near and far bounds in here too, if we want
            self.data = torch.cat((self.all_rays[:,:6],self.all_rgbs[:,:]),1)

    def process_frame(self, frame):
        pose = np.array(frame['transform_matrix'])[:3, :4]
        self.poses += [pose]
        c2w = torch.FloatTensor(pose)
        image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
        self.image_paths += [image_path]
        img = Image.open(image_path)
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#PIL.Image.LANCZOS
        img = img.resize(self.img_wh, Image.LANCZOS) # lanczos is best for downsampling
        img = self.transform(img) # (4, h, w)
        if self.n_chan == 4:
            assert img.shape == (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
        elif self.n_chan == 3: # no alpha
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
        elif self.n_chan == 1:
            img = img.expand(3, -1, -1).view(3,-1).permute(1,0) # (h*w, 3) RGB
        noise = Image.effect_noise(self.img_wh, self.noise_sd)
        noise = self.transform(noise)
        noise -= 0.5 # noise centred at 0.5
        noise = noise.expand(3,-1,-1).view(3, -1).permute(1, 0)

        img = (img + self.noise_level*noise).clamp(0,1)

        # ray direction is normalised
        rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
        return rays_o, rays_d, img


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test':
            return 1

    def __getitem__(self, idx):
        if (self.split == 'train'):
            return self.data[idx]
        if ((self.split == 'test') or (self.split == 'video')):
            rays_o, rays_d, img = self.process_frame(self.meta['frames'][idx])
            rays = torch.cat([rays_o, rays_d, img],1)
            return rays