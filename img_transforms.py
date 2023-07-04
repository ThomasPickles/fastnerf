import numpy as np
from skimage import exposure, transform # supports 16-bit tiff

def walnut(img, shape):
    # Walnut data needs to be transformed 5 pixel to the left
    img = np.roll(img, -5, axis=1)
    img = -np.log(img)
    img = transform.resize(img, shape, anti_aliasing=True)
    # subtract off background values
    background = 0.25*(img[0,0]+img[shape[0]-1,0]+img[shape[0]-1,shape[1]-1]+img[0,shape[1]-1])
    img -= background
    # CLAMP BACKGROUND VALUES TO ZERO
    # nerf is overfitting to noise, so clamp
    # any shot noise to zero by zeroing small values
    dark_floor = 0.05 # determine empirically
    img = exposure.rescale_intensity(img, in_range=(dark_floor, 1.8), out_range=(0.,1.))
    return img

def jaw(img, shape):
    img = transform.resize(img, shape, anti_aliasing=True)
    # img = exposure.rescale_intensity(img, in_range=(0, 2.**6), out_range=(0.,1.))
    img = np.float32(img)
    return img

