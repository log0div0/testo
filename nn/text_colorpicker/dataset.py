
import math
import torch
import os
from PIL import Image
import numpy as np
import json
import visdom
import random
import argparse
from .model import CHAR_WIDTH, CHAR_HEIGHT, colors

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
args = parser.parse_args()

vis = None

class Dataset:
	def __init__(self, root, size):
		self.root = root
		self.size = size
		self.cache = [None] * size

	def __len__(self):
		return self.size

	def make_char(self, image, char):
		left = math.floor(char['bbox']['left'])
		top = math.floor(char['bbox']['top'])
		right = math.ceil(char['bbox']['right']) + 1
		bottom = math.ceil(char['bbox']['bottom']) + 1

		width = right - left
		height = bottom - top

		left -= random.randint(0, width // 4)
		right += random.randint(0, width // 4)
		top -= random.randint(0, height // 4)
		bottom += random.randint(0, height // 4)

		if left < 0:
			left = 0
		if top < 0:
			top = 0
		if right > image.shape[1]:
			right = image.shape[1]
		if bottom > image.shape[0]:
			bottom = image.shape[0]

		resample = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])

		char_img = image[top:bottom, left:right]
		char_img = Image.fromarray(char_img)
		char_img = char_img.resize([CHAR_WIDTH, CHAR_HEIGHT], resample)
		char_img = np.array(char_img)
		char_img = np.transpose(char_img, [2, 0, 1])

		# vis = visdom.Visdom()
		# vis.image(char_img)
		# vis.text(char['color'])
		# vis.text(char['backgroundColor'])
		# exit()

		char_img = torch.from_numpy(char_img).float() / 255.0

		char_color = colors.index(char['color'])
		if char_color < 0:
			raise Exception("unknown color: " + char['color'])

		char_background_color = colors.index(char['backgroundColor'])
		if char_background_color < 0:
			raise Exception("unknown color: " + char['backgroundColor'])

		return {
			'img': char_img,
			'color': char_color,
			'background_color': char_background_color
		}

	def _load_data(self, idx):
		image_path = os.path.join(self.root, str(idx) + '.png')
		label_path = os.path.join(self.root, str(idx) + '.json')

		image = Image.open(image_path)
		image = np.array(image)
		image = image[:,:,:3]

		with open(label_path) as f:
			label = json.loads(f.read())

		chars = []
		for textline in label['textlines']:
			for char in textline['chars']:
				if char['text'] != ' ':
					chars.append(self.make_char(image, char))

		return chars

	def load_data(self, idx):
		if not self.cache[idx]:
			self.cache[idx] = self._load_data(idx)
		return self.cache[idx]

	def __getitem__(self, idx):
		return random.choice(self.load_data(idx))

datasets = []

for root, dirs, files in os.walk(args.dataset_folder):
	labels_count = len([file for file in files if os.path.splitext(file)[1] == '.json'])
	if not labels_count:
		continue
	datasets.append(Dataset(root, labels_count))

dataset = torch.utils.data.ConcatDataset(datasets)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == '__main__':
	for data in data_loader:
		print(data['img'].shape, data['color'].shape, data['background_color'].shape)
		exit(0)
