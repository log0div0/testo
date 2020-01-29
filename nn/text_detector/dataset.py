
import math
import torch
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import json
import visdom
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
args = parser.parse_args()

class Dataset:
	def __init__(self, root, size):
		self.root = root
		self.size = size
		self.cache = [None] * size

	def __len__(self):
		return self.size

	def _load_data(self, idx):
		image_path = os.path.join(self.root, str(idx) + '.png')
		label_path = os.path.join(self.root, str(idx) + '.json')

		image = Image.open(image_path)
		image = np.array(image)
		image = image[:,:,:3]
		image = np.transpose(image, [2, 0, 1])

		_, image_height, image_width = image.shape

		with open(label_path) as f:
			label = json.loads(f.read())

		up_image = np.zeros([image_height, image_width], dtype=np.uint32)
		down_image = np.zeros([image_height, image_width], dtype=np.uint32)

		for textline in label['textlines']:
			up_bboxes = []
			down_bboxes = []
			for char in textline['chars']:
				if char['text'] == ' ':
					continue
				left = math.floor(char['bbox']['left'])
				top = math.floor(char['bbox']['top'])
				right = math.ceil(char['bbox']['right']) + 1
				bottom = math.ceil(char['bbox']['bottom']) + 1
				center = math.floor((top + bottom) / 2) + 1
				up_bboxes.append([left, top, right, center])
				down_bboxes.append([left, center, right, bottom])
			up_image = self.draw(up_image, up_bboxes)
			down_image = self.draw(down_image, down_bboxes)

		up_image = np.clip(up_image, 0, 255).astype(np.uint8)
		down_image = np.clip(down_image, 0, 255).astype(np.uint8)

		# label_image = np.stack([up_image, down_image, down_image], axis=0)
		# vis = visdom.Visdom()
		# vis.image(image)
		# vis.image(label_image)
		# exit()

		return image, up_image, down_image

	def draw(self, image, bboxes):
		tmp_image = Image.new('L', (image.shape[1], image.shape[0]))
		tmp_draw = ImageDraw.Draw(tmp_image)

		min_height = 0

		for bbox in bboxes:
			left, top, right, bottom = bbox
			height = bottom - top

			if min_height < height:
				min_height = height

			tmp_draw.rectangle(bbox, fill=255)

		blur_radius = min_height // 3
		if blur_radius > 1:
			tmp_image = tmp_image.filter(ImageFilter.BoxBlur(blur_radius))

		return image + np.array(tmp_image)

	def load_data(self, idx):
		if not self.cache[idx]:
			self.cache[idx] = self._load_data(idx)
		return self.cache[idx]

	def __getitem__(self, idx):
		image, up_image, down_image = self.load_data(idx)

		c, h, w = image.shape
		crop_h = 480
		crop_w = 640
		y = random.randint(0, h - crop_h)
		x = random.randint(0, w - crop_w)
		image = image[:, y:y + crop_h, x:x + crop_w]
		up_image = up_image[y:y + crop_h, x:x + crop_w]
		down_image = down_image[y:y + crop_h, x:x + crop_w]

		image = torch.from_numpy(image).float() / 255
		up_image = torch.from_numpy(up_image).float() / 255
		down_image = torch.from_numpy(down_image).float() / 255

		return image, up_image, down_image

datasets = []

for root, dirs, files in os.walk(args.dataset_folder):
	labels_count = len([file for file in files if os.path.splitext(file)[1] == '.json'])
	if not labels_count:
		continue
	datasets.append(Dataset(root, labels_count))

dataset = torch.utils.data.ConcatDataset(datasets)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

if __name__ == '__main__':
	for data in data_loader:
		print(data[0].shape, data[1].shape, data[2].shape)
		exit(0)
