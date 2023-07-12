import numpy as np
from datasets import get_params

# phantom is centred at 0,0,0
# voxel_size = 0.3
# swapping axes (x,y,z) -> (3,-2,-1)
def world_to_local(points):
    ''' Takes an array of size [n_points, 3] '''
    n_points =  points.shape[0]
    assert points.shape[1] == 3, 'invalid size'
    expanded_points = np.concatenate((points, np.ones((n_points,1))), axis=1)
    # We only have ground truth data for jaw dataset
    _, radius, object_size, _ = get_params("jaw")
    h_offset = 331 / 2
    w_offset = 275 / 2
    scale = 2 * object_size / 0.3  # effective voxel size 

    # map 0.5 to 275/2
    # map 0 to -200 ish
    # map 0 to 500 or so
    M = np.array([
        [0,     0,    -scale, h_offset + scale / 2.],
        [0,    -scale, 0,     w_offset + scale / 2.],
        [scale, 0,     0,     w_offset - scale / 2.]
        ])
    M = np.expand_dims(M, 0)
    expanded_points = np.expand_dims(expanded_points, 2)
    return np.matmul(M,expanded_points).squeeze()

def in_bounds(x, upper):
    # we use zero indexing, so x needs to be explicitly
    # less than the phantom size
    dimension_in_bounds = (x < upper) * (x >= 0)
    all_in_bounds = np.all(dimension_in_bounds, axis = 1, keepdims=True)
    return all_in_bounds

def get_value_from_phantom(vox, phantom):
    ''' Safe method which does bounds checking '''
    bounds = np.array(phantom.shape)
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

if __name__ == '__main__':

    phantom = np.load('jaw/jaw_phantom.npy')
    h,w,_ = phantom.shape

    ## TESTING ###
    world = np.array([
        [0.5,0.5,0.5], # middle of phantom
        [0.6,0.6,0.6], # middle of phantom
        [0, 0, 0], # empty
        [1, 1, 1], # empty
    ])
    sigmas = get_sigma_gt(world, phantom)
    print(sigmas)
    assert sigmas[0] == 0, 'windpipe'
    assert sigmas[1] > 0, 'tissue'
    assert sigmas[2] == 0, 'out of phantom'
    assert sigmas[3] == 0, 'out of phantom'




