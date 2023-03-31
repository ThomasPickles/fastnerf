import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

def view_training_curve(f):

	with open(f, 'r') as trainingfile:
		data = json.load(trainingfile)

	timestamp = data["timestamp"]
	model = data["model"]
	final_training_loss_db = data["final_training_loss_db"]
	epochs = data["epochs"]
	img_size = data["img_size"]
	rendering = data["rendering"]
	samples = data["samples"]
	batchsize = data["batchsize"]
	run_time = data["run_time"]
	device = data["device"]
	checkpoint = data["checkpoint"]
	training_loss = data["training_loss"]

	plt.style.use('_mpl-gallery')

	# plot
	fig, ax = plt.subplots()

	ax.plot(training_loss, linewidth=2.0)

	ax.set(xlabel='Batch no', ylabel='Training loss (dB)',
		title=f'Progression of loss function for img_size {img_size}')
	ax.grid()
	ax.invert_yaxis()

	plt.show()

def parse_args():
	parser = argparse.ArgumentParser(description="Train model")
	parser.add_argument("filename")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	view_training_curve(args.filename)