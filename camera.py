import matplotlib.pyplot as plt
import numpy as np
from convert_data import BlenderDataset

def plot_cameras():
    
    W, H = 275, 331
    n_train, n_test = 14,4
    n = n_train + n_test
    rays = np.zeros((2*n,4))
    for split in ['train','test']:
        data = BlenderDataset('jaw', split=split, n_chan=3)
        n_max = n_train if split == 'train' else n_test
        for frame in range(n_max):
            ray_o = data[frame * H * W, :2]
            ray_d = data[frame * H * W:(frame +1) * H * W, 3:5]
            theta = np.arctan(ray_d[:,1]/ray_d[:,0])
            t_max, t_min = np.argmax(theta), np.argmin(theta)
            print(f"min and max angles are {theta[t_min]} and {theta[t_max]}")
            img_index = frame if split == 'train' else frame + n_train
            rays[2*img_index,:2] = ray_o
            rays[2*img_index, 2:4] = 300*ray_d[t_min,:]
            rays[2*img_index+1,:2] = ray_o
            rays[2*img_index+1, 2:4] = 300*ray_d[t_max,:]

    print(rays.shape)
    print(rays)
    plt.style.use('_mpl-gallery-nogrid')
  
    fig, ax = plt.subplots()

    ax.quiver(rays[:2*n_train,0], rays[:2*n_train,1], rays[:2*n_train,2], rays[:2*n_train,3], color="C0", angles='uv',scale_units='xy', scale=5, width=.005)
    ax.quiver(rays[2*n_train:,0], rays[2*n_train:,1], rays[2*n_train:,2], rays[2*n_train:,3], color="C1", angles='uv',scale_units='xy', scale=5, width=.005)


    plt.show()

plot_cameras()

