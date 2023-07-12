# JSON Configuration Documentation

This document lists the JSON parameters of all components of configuration files.  An example config file is available at [[config/test_config.json]].

For each component, I provide a sample configuration that lists each parameter's default value.

__Network__ and __Encoding__ config options are inherited from tiny cuda [documentation](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md)


### Optimisation

```json5
{
        "training_epochs": 10,      // Number of training passes over all data
        "loss": "L2",               // Currently only L2 is supported 
        "learning_rate": 0.001,     // Initial stepsize
        "milestones": [2,4,6,8],    // Stepsize modified at these epochs
        "gamma": 0.7,               // Factor to multiply stepsize
        "batchsize": 1024,          // Number of training steps before gradient descent step
        "samples_per_ray": 192,     // Number of samples to take along each ray
        "pixel_importance_sampling": false,     // Should pixels be sampled proportionally to intensity value
        "random_seed": false,                   // Seed random number generator for reproducibility
}
```

### Data

```json5
{
        "dataset": "jaw",                   // Sets search path for image files.  ./data/{dataset}.  Currently "jaw" or "walnut"
        "transforms_file": "transforms",    // Base name of transforms file.  Found in data/{dataset}
        "img_size": 50,                     // Long dimension of training images
        "n_images": 50,                     // Number of images to train on. 
        "noise_level": 0.0,                 // Noise level to add to training images.  Does not affect test data
    }
```
    
### Output

```json5
{
        "images": true,
        "slices": true,
        "samples_per_ray": 192,
        "slice_resolution": 400,
        "rays_per_pixel": 16,
        "interval": 10,
        "intermediate_slices": false,
        "path": "hash"
    }
```

### Hardware

```json5
{
        "train": "cuda",
        "test": "cuda"
    }
```
