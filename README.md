# NeRF for Medical Imaging

This is a NeRF library that has been tested on medical images.  Medical images have differences from the dataset that NeRF is typically trained on:
- No colour information
- No specular highlights
- Volumetric rendering of images follows different process 

# Installation

Install tiny-cuda-nn:
https://github.com/NVlabs/tiny-cuda-nn
Ensure that you have got the [PyTorch extension](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) working.

The code falls back to a standard PyTorch implementation, but is significantly slower without this extension.

Set up a conda environment and install the required packages:
```sh
conda create --name nerf --file conda-requirements.txt
conda install --name nerf --file conda-requirements.txt
```

Download walnut dataset from [this webpage](https://zenodo.org/record/6986012).  The dataset is large, totalling around 4Gb.  The tiff files should be stored in ./data/walnut/

# Execution

```sh
conda activate nerf
python main.py [config_file]
```

Example config file in config/test_config.json
Should take about 1 minute to train for 40 epochs
results should be similar to /img/test_config

Training curves and slices will be output to out/{run_name}/

## Config file

[The JSON documentation](DOCUMENTATION.md) lists configuration options.

## TODO
- fix bug with jaw data
- add some .png files to img/ folder to show example output
- modify main.py to output info in out.csv
