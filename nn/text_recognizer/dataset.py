
import math
import torch
import os
import PIL.Image
import numpy as np
import json
import visdom
import random
import argparse
from .model import MAX_HEIGHT

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

	def make_word(self, image, chars):
		word_str = ''.join([char['text'] for char in chars])

		left = math.floor(min([char['bbox']['left'] for char in chars]))
		right = math.ceil(max([char['bbox']['right'] for char in chars])) + 1
		top = math.floor(min([char['bbox']['top'] for char in chars]))
		bottom = math.ceil(max([char['bbox']['bottom'] for char in chars])) + 1

		width = right - left
		height = bottom - top

		left -= random.randint(-1, height // 4)
		right += random.randint(-1, height // 4)
		top -= random.randint(-1, height // 4)
		bottom += random.randint(-1, height // 4)

		if left < 0:
			left = 0
		if right > image.shape[1]:
			right = image.shape[1]
		if top < 0:
			top = 0
		if bottom > image.shape[0]:
			bottom = image.shape[0]

		width = right - left
		height = bottom - top

		word_img = image[top:bottom, left:right]

		ratio = width / height
		height = MAX_HEIGHT
		width = math.floor(ratio * MAX_HEIGHT)
		if width % 2:
			width += 1
		width *= 2

		resample = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])

		word_img = PIL.Image.fromarray(word_img)
		word_img = word_img.resize([width, height], resample)
		word_img = np.array(word_img)
		word_img = np.transpose(word_img, [2, 0, 1])

		# vis = visdom.Visdom()
		# vis.image(word_img)
		# vis.text(word_str)
		# exit()

		word_img = torch.from_numpy(word_img).float() / 255.0

		return {
			'str': word_str,
			'img': word_img
		}

	def _load_data(self, idx):
		image_path = os.path.join(self.root, str(idx) + '.png')
		label_path = os.path.join(self.root, str(idx) + '.json')

		image = PIL.Image.open(image_path)
		image = np.array(image)
		image = image[:,:,:3]

		with open(label_path) as f:
			label = json.loads(f.read())

		words = []
		chars = []
		for textline in label['textlines']:
			for char in textline['chars']:
				if char['text'] == ' ':
					if len(chars):
						words.append(self.make_word(image, chars))
						chars = []
					continue
				chars.append(char)
			if len(chars):
				words.append(self.make_word(image, chars))
				chars = []

		return words

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

def collate(words):
	images_size = [word['img'].shape[-1] for word in words]
	max_width = max(images_size)
	images = torch.zeros([len(words), 3, MAX_HEIGHT, max_width])
	for i, word in enumerate(words):
		images[i, :, :, :word['img'].shape[-1]] = word['img']
	labels = [word['str'] for word in words]
	return images, labels, torch.Tensor(images_size).long()

data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

if __name__ == '__main__':
	for data in data_loader:
		print(data[0].shape, data[1], data[2])
		exit(0)
