#!/usr/bin/env python3

import argparse
import torch
import numpy as np

import convert_data as conv

def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

	parser.add_argument("--file", default=".", help="The root dir.")

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	f = args.file
	d = conv.BlenderDataset(f)
	d.define_transforms()
	print(d.transform)