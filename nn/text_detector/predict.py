
import os
import math
import argparse
import numpy as np
from PIL import Image
import onnxruntime

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output_folder', default='output')
args = parser.parse_args()

def list_files(in_path):
	img_files = []
	for (dirpath, dirnames, filenames) in os.walk(in_path):
		for file in filenames:
			filename, ext = os.path.splitext(file)
			if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
				img_files.append(os.path.join(dirpath, file))
	return img_files

def is_n_times_div_by_2(value, n=4):
	for _ in range(n):
		if value % 2 != 0:
			return False
		value /= 2
	return True

def nearest_n_times_div_by_2(value, n=4):
	while True:
		if is_n_times_div_by_2(value, n):
			return value
		value += 1

def pad_image(image):
	c, h, w = image.shape
	new_h = nearest_n_times_div_by_2(h)
	new_w = nearest_n_times_div_by_2(w)
	new_image = np.zeros([c, new_h, new_w], dtype=np.float32)
	new_image[:, :h, :w] = image
	return new_image

ort_session = onnxruntime.InferenceSession(args.model)

if not os.path.isdir(args.output_folder):
	os.mkdir(args.output_folder)

for image_path in list_files(args.input_folder):
	print(image_path)
	image = Image.open(image_path)
	image = np.array(image)
	image = image[:,:,:3]
	h, w, c = image.shape

	x = image / 255
	x = x.transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
	x = pad_image(x)
	x = x[np.newaxis]                # [c, h, w] to [b, c, h, w]

	ort_inputs = {"input": x}
	ort_outs = ort_session.run(None, ort_inputs)
	y = ort_outs[0][0]

	up_image = y[0,:h,:w]
	down_image = y[1,:h,:w]

	up_image = (np.clip(up_image, 0, 1) * 255).astype(np.uint8)
	down_image = (np.clip(down_image, 0, 1) * 255).astype(np.uint8)
	res_img = np.hstack((image,np.stack([up_image, down_image, down_image], axis=2)))

	filename, file_ext = os.path.splitext(os.path.basename(image_path))
	res_img_file = os.path.join(args.output_folder, filename + '.png')

	Image.fromarray(res_img).save(res_img_file)

	del x
	del y

print("OK")
