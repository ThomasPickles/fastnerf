# from scipy.interpolate import interpn
from skimage import io

import numpy as np

class NerfImage():
    def __init__(self, path, img_transform, im_wh):
        numpy_image = io.imread(path)
        dtype = numpy_image.dtype
        assert (dtype == 'uint8') or (dtype == 'uint16'), 'unknown datatype'
        if numpy_image.ndim == 3:
            numpy_image = numpy_image[:,:,1] # only keep the first colour channel if RGB
        h, w = numpy_image.shape 
        assert h > w, 'data might be wrong way round!'
        img = img_transform(numpy_image, im_wh)

        # rescale
        self.image = img
        self.h, self.w = self.image.shape
        assert self.h > self.w, 'data might be wrong way round!'

    def get_pixel_normalised(self, x, y):
        # read 5 pixels to right (circular)
        assert 0. <= x <= 1., 'img coords must be normalised'
        assert 0. <= y <= 1., 'img coords must be normalised'
        x *= self.w
        y *= self.h
        return self.image[int(y), int(x)]

if __name__ == '__main__':
    path = 'walnut/20201111_walnut_0350.tif'
    transform = lambda img: 5-np.log(img / 2**12)
    img = NerfImage(path, transform)
    print(img.get_pixel_normalised(0.5, 0.5))
    print(img.get_pixel_normalised(0.4, 0.4))
    print(img.get_pixel_normalised(0.25, 0.25))
    print(img.get_pixel_normalised(0, 0)) # black
    img.get_pixel_normalised(-1, -1) # error