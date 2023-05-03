import numpy as np
import matplotlib.pyplot as plt

# height_detector = 331         # = number of pixel
# width_detector = 275          # = number of pixel
# source_origin = 401.069       # distance between the X-ray source and center of rotation, ie. center of the object, in mm
# origin_det = 163.233          # distance between the object and the detector, in mm

# near = 163 - 60 # camera is on a circle 163 from origin at (x,y,0)
# far = 163 + 60 # 50 is approx


# phantom is centred at 0,0,0
# voxel_size = 0.3
# swapping axes (x,y,z) -> (3,-2,-1)
def world_to_local(points):
    ''' Takes an array of size [n_points, 3] '''
    n_points =  points.shape[0]
    assert points.shape[1] == 3, 'invalid size'
    expanded_points = np.concatenate((points,np.ones((n_points,1))),axis=1)
    M = np.array([
        [0,0,-1/0.3,331/2],
        [0,-1/0.3,0,275/2],
        [1/0.3,0,0,275/2]
        ])
    M = np.expand_dims(M, axis=0)
    expanded_points = np.expand_dims(expanded_points, axis=2)
    return np.matmul(M,expanded_points).squeeze()

# swapping axes (1,2,3) -> (-z,-y,x)
def local_to_world(voxs):
    ''' Takes an array of size [n_voxs, 3] '''
    n_voxs =  voxs.shape[0]
    assert voxs.shape[1] == 3, 'invalid size'
    trans_vox = np.concatenate((voxs,np.ones((n_voxs,1))),axis=1)
    M = np.array([
        [0,0,1*0.3,-0.3*275/2],
        [0,-1*0.3,0,0.3*275/2],
        [-1*0.3,0,0,0.3*331/2]
        ])
    M = np.expand_dims(M, axis=0)
    trans_vox = np.expand_dims(trans_vox, axis=2)
    return np.matmul(M,trans_vox).squeeze()


def in_bounds(x, upper):
    # we use zero indexing, so x needs to be explicitly
    # less than the phantom size
    dimension_in_bounds = (x < upper) * (x >= 0)
    all_in_bounds = np.all(dimension_in_bounds, axis = 1, keepdims=True)
    return all_in_bounds

def get_value_from_phantom(vox, phantom):
    ''' Safe method which does bounds checking '''
    bounds = np.asarray(phantom.shape)
    bounded_vx = np.clip(vox, 0.5, bounds-0.5)
    vx = np.round(bounded_vx).astype(int)
    values = phantom[vx[:,0], vx[:,1], vx[:,2]]
    return values

def get_sigma_gt(points, phantom):
    assert points.shape[1] == 3, 'needs to be 3d points'
    vx = world_to_local(points)
    # HACK: just taking the lower value is a bit of a
    # hack.  we could definitely do some interpolation here
    bounds = np.asarray(phantom.shape)
    in_points = in_bounds(vx, bounds)

    sigma = get_value_from_phantom(vx, phantom)
    # although points clipped to the boundary are probably
    # zero density anyway, it's safer to explicitly set
    # them to zero
    sigma = np.where(in_points, np.expand_dims(sigma, axis=1), 0)
    return sigma

phantom = np.load('jaw/jaw_phantom.npy')
h,w,_ = phantom.shape



## TESTING ###

world_data = [
    [-160, 0, 0], # frame 0 test, from right, ray origin
    [-50, 155, 0], # frame 1 test, straight on, ray origin
    [145,-70,0], # frame 2 test, from left, ray origin
    [0, 0, 0], # origin
    [0, 0, 1] # tmp
    ]
world_coords = np.asarray(world_data)
lc_out = world_to_local(world_coords)
assert lc_out[0,2] < 0, 'axis 3 should be too small'
assert lc_out[1,1] < 0, 'axis 2 should be too small'
assert lc_out[2,2] > w, 'axis 3 should be too large'

local_data = [
    [128,99,176], # 2 teeth
    [128,99,98], # 1 tooth
    [0, 0, 0], # origin
    [5, 5, 5],
    [h-1,w-1,w-1] # furthest voxel
    # []
]
local_coords = np.asarray(local_data)
wc_out = local_to_world(local_coords)
sigmas = get_value_from_phantom(local_coords, phantom)
assert sigmas[0] == 1.0, 'tooth'
assert sigmas[1] == 1.0, 'tooth'
assert sigmas[2] == 0.0, 'empty space'
assert sigmas[3] == 0.0, 'empty space'
# assert sigmas[4] == 0.1, 'spine'
# assert sigmas[5] == 0.05, 'tissue'



