import numpy as np
import matplotlib.pyplot as plt

def points_to_voxels(points, resolution):
    resolution = np.expand_dims(resolution, 0)
    return points + resolution/2

def get_sigma_gt(points, phantom):
    assert points.shape[1] == 3, 'needs to be 3d points'
    vx = points_to_voxels(points, np.asarray(phantom.shape))
    vx_low = np.floor(vx).astype(int)
    return phantom[vx_low[:,0],vx_low[:,1],vx_low[:,2]]
