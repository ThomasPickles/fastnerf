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
def points_to_voxels(points, bounds):
    bounds = np.expand_dims(bounds, 0)
    points[:,[0,1,2]] = points[:,[2,1,0]]
    vx = points/0.3 + bounds/2
    return vx

def in_bounds(x, upper):
    # we use zero indexing, so x needs to be explicitly
    # less than the phantom size
    dimension_in_bounds = (x < upper) * (x >= 0)
    all_in_bounds = np.all(dimension_in_bounds, axis = 1, keepdims=True)
    return all_in_bounds

def get_value_from_phantom(bounded_vx, phantom):
    # phantom is z,x,y
    # points are 
    values = phantom[bounded_vx[:,0],bounded_vx[:,1],bounded_vx[:,2]]
    return np.expand_dims(values, axis=1)

def get_sigma_gt(points, phantom):
    assert points.shape[1] == 3, 'needs to be 3d points'
    bounds = np.asarray(phantom.shape)
    vx = points_to_voxels(points, bounds)
    # HACK: just taking the lower value is a bit of a 
    # hack.  we could definitely do some interpolation here
    vx_int = np.floor(vx).astype(int) # [200,3]
    in_points = in_bounds(vx_int, bounds)
    # make sure we clip indices so that we
    # don't get out-of-bounds errors
    bounded_vx = np.clip(vx_int,0, bounds-1)
    sigma = get_value_from_phantom(bounded_vx, phantom)
    # although points clipped to the boundary are probably
    # zero density anyway, it's safer to explicitly set
    # them to zero
    sigma = np.where(in_points, sigma, 0)
    return sigma

