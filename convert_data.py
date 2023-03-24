import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from torchvision import io

from ray_utils import *

class BlenderDataset(Dataset):
    # TODO: NEED TO SORT WIDTH AND HEIGHT...
    def __init__(self, root_dir, split='train', img_wh=(275, 331), n_chan=4):
        self.root_dir = root_dir
        self.split = split
        self.n_chan = n_chan
        # FIXME: why must it be square?
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        if (self.split == 'train' or self.split == 'test'): # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            i = 0
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                # FIXME: this resize operation is not so good for non-square
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                if self.n_chan == 4:
                    assert img.shape == (4, h, w)
                    img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                elif self.n_chan == 3: # no alpha
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                if i == 0:
                    m0 = int(h/2*w)
                    mw = int((h/2+1)*w)
                    print(f"middle row is {img[m0:mw,:]}")
                
                self.all_rgbs += [img]
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                i += 1

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            assert torch.max(self.all_rgbs) <= 1.0 # should all be between 0 and 1
            self.data = torch.cat((self.all_rays[:,:6],self.all_rgbs[:,:]),1)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        return self.data[idx]