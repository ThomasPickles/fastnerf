import imageio
from skimage import io
from skimage import exposure, transform # supports 16-bit tiff

import numpy as np

class NerfImage():
    def __init__(self, path, img_transform):
        numpy_image = io.imread(path)
        dtype = numpy_image.dtype
        h, w = numpy_image.shape 
        # walnut images are misaligned by 5 pixels.  correct it here
        numpy_image = np.roll(numpy_image, 5, axis=0) # TODO: check this
        img = img_transform(numpy_image)
        background = 0.25*(img[0,0]+img[h-1,0]+img[h-1,w-1]+img[0,w-1])
        img -= background
        # clamp background values to zero
        img = exposure.rescale_intensity(img, in_range=(0.1, 1.8), out_range=(0.,1.))
        print("min, max (after):", img.min(), img.max())
        img = transform.rescale(img, 0.1, anti_aliasing=True)
        # TODO: if we want to decrease res, do it here
        # print(self.image)
        # put inside a pillow wrapper to take advantage of interpolation
        self.image = img
        self.h, self.w = self.image.shape
        print(f"img height, width: {self.h}, {self.w}")

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