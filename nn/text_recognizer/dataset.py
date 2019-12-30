
import math
import torch
import os
import PIL.Image
import numpy as np
import json
import visdom
import random
from .crnn import MAX_HEIGHT

class Dataset:
	def __init__(self, root, size):
		self.root = root
		self.size = size
		self.cache = [None] * size

	def __len__(self):
		return self.size

	def make_word(self, image, chars):
		word_str = ''.join([char['text'] for char in chars])

		left = math.floor(chars[0]['bbox']['left'])
		right = math.ceil(chars[-1]['bbox']['right']) + 1
		top = math.floor(min([char['bbox']['top'] for char in chars]))
		bottom = math.ceil(max([char['bbox']['bottom'] for char in chars])) + 1

		width = right - left
		height = bottom - top

		left -= random.randint(-1, int(chars[0]['bbox']['width']))
		right += random.randint(-1, int(chars[-1]['bbox']['width']))
		top -= random.randint(-1, height // 4)
		bottom += random.randint(-1, height // 4)

		word_img = image[top:bottom, left:right]

		ratio = width / height
		height = MAX_HEIGHT
		width = int(ratio * height)

		word_img = PIL.Image.fromarray(word_img)
		word_img = word_img.resize([width, height], PIL.Image.BILINEAR)
		word_img = np.array(word_img, np.float32)
		word_img /= 255.0
		word_img -= 0.5

		word_img = torch.from_numpy(word_img).unsqueeze(0)

		return {
			'str': word_str,
			'img': word_img
		}

	def _load_data(self, idx):
		image_path = os.path.join(self.root, str(idx) + '.png')
		label_path = os.path.join(self.root, str(idx) + '.json')

		image = PIL.Image.open(image_path)
		image = image.convert("L")
		image = np.array(image)

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
		words = self.load_data(idx)
		word = random.choice(words)

		# vis = visdom.Visdom()
		# vis.image(np.transpose(word['img'], [2, 0, 1]))
		# vis.text(word['str'])
		# exit()

		return word

datasets = []

for root, dirs, files in os.walk('dataset'):
	labels_count = len([file for file in files if os.path.splitext(file)[1] == '.json'])
	if not labels_count:
		continue
	datasets.append(Dataset(root, labels_count))

dataset = torch.utils.data.ConcatDataset(datasets)

def collate(words):
	max_width = max([word['img'].shape[-1] for word in words])
	images = torch.zeros([len(words), 1, MAX_HEIGHT, max_width])
	for i, word in enumerate(words):
		images[i, :, :, :word['img'].shape[-1]] = word['img']
	labels = [word['str'] for word in words]
	images_widths = [word['img'].shape[-1] for word in words]
	return images, labels, torch.Tensor(images_widths)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)

if __name__ == '__main__':
	for data in data_loader:
		print(data[0].shape, data[1])
		exit(0)
