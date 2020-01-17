
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

		image_height, image_width, _ = image.shape

		with open(label_path) as f:
			label = json.loads(f.read())

		textlines = []
		words = []
		chars = []
		for textline in label['textlines']:
			space_count = 0
			for char in textline['chars']:
				if char['text'] == ' ':
					if len(chars):
						words.append(chars)
						chars = []
					space_count += 1
					if space_count > 1:
						if len(words):
							textlines.append(words)
							words = []
				else:
					left = math.floor(char['bbox']['left'])
					top = math.floor(char['bbox']['top'])
					right = math.ceil(char['bbox']['right']) + 1
					bottom = math.ceil(char['bbox']['bottom']) + 1
					chars.append([left, top, right, bottom])
					space_count = 0
			if len(chars):
				words.append(chars)
				chars = []
			if len(words):
				textlines.append(words)
				words = []

		region_image = Image.new('L', [image_width, image_height])

		textlines_bboxes = []
		for words in textlines:
			words_bboxes = []
			for chars in words:
				left = chars[0][0]
				top = chars[0][1]
				right = chars[0][2]
				bottom = chars[0][3]
				for char in chars[1:]:
					left = min(left, char[0])
					top = min(top, char[1])
					right = max(right, char[2])
					bottom = max(bottom, char[3])
				words_bboxes.append([left, top, right, bottom])
			textlines_bboxes.append(words_bboxes)

		for words_bboxes in textlines_bboxes:
			for i in range(len(words_bboxes)):
				region_image = self.draw(region_image, words_bboxes[i])

		return image, np.array(region_image)

	def draw(self, image, bbox):
		left, top, right, bottom = bbox
		width = right - left
		height = bottom - top
		blur_radius = min(width, height) // 2
		tmp_image = Image.new('L', image.size)
		tmp_draw = ImageDraw.Draw(tmp_image)
		tmp_draw.rectangle(bbox, fill=255)
		tmp_image = tmp_image.filter(ImageFilter.BoxBlur(blur_radius))
		return Image.composite(image, tmp_image, ImageOps.invert(tmp_image))

	def load_data(self, idx):
		if not self.cache[idx]:
			self.cache[idx] = self._load_data(idx)
		return self.cache[idx]

	def __getitem__(self, idx):
		image, region_image = self.load_data(idx)

		image = np.transpose(image, [2, 0, 1])

		# vis = visdom.Visdom()
		# vis.image(image)
		# vis.image(region_image)
		# exit()

		c, h, w = image.shape
		crop_h = 480
		crop_w = 640
		y = random.randint(0, h - crop_h)
		x = random.randint(0, w - crop_w)
		image = image[:, y:y + crop_h, x:x + crop_w]
		region_image = region_image[y:y + crop_h, x:x + crop_w]

		image = torch.from_numpy(image).float() / 255
		region_image = torch.from_numpy(region_image).float() / 255

		return image, region_image

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
