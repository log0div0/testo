
import math
import cv2
import torch
import os
import PIL.Image
import numpy as np
import json
import visdom
from .craft import normalizeMeanVariance
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder')
args = parser.parse_args()

class GaussianTransformer(object):
	def __init__(self, imgSize=512, region_threshold=0.4,
				 affinity_threshold=0.2):
		distanceRatio = 3.34
		scaledGaussian = lambda x: math.exp(-(1 / 2) * (x ** 2))
		self.region_threshold = region_threshold
		self.imgSize = imgSize
		self.standardGaussianHeat = self._gen_gaussian_heatmap(imgSize, distanceRatio)

		_, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
		np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
		x, y, w, h = cv2.boundingRect(np_contours)
		self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
		_, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
		np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
		x, y, w, h = cv2.boundingRect(np_contours)
		self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
		self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
							   dtype=np.int32)

	def _gen_gaussian_heatmap(self, imgSize, distanceRatio):
		scaledGaussian = lambda x: math.exp(-(1 / 2) * (x ** 2))
		heat = np.zeros((imgSize, imgSize), np.uint8)
		for i in range(imgSize):
			for j in range(imgSize):
				distanceFromCenter = np.linalg.norm(np.array([i - imgSize / 2, j - imgSize / 2]))
				distanceFromCenter = distanceRatio * distanceFromCenter / (imgSize / 2)
				scaledGaussianProb = scaledGaussian(distanceFromCenter)
				heat[i, j] = np.clip(scaledGaussianProb * 255, 0, 255)
		return heat

	def add_region_character(self, image, target_bbox, regionbox=None):

		if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
				target_bbox[:, 1] > image.shape[0]):
			return image
		affi = False
		if regionbox is None:
			regionbox = self.regionbox.copy()
		else:
			affi = True

		M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
		oribox = np.array(
			[[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
			dtype=np.float32)
		test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
		real_target_box = cv2.perspectiveTransform(oribox, M)[0]
		real_target_box = np.int32(real_target_box)
		real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
		real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

		if np.any(target_bbox[0] < real_target_box[0]) or (
				target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
				target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
				target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
			# if False:
			warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
			warped = np.array(warped, np.uint8)
			image = np.where(warped > image, warped, image)
		else:
			xmin = real_target_box[:, 0].min()
			xmax = real_target_box[:, 0].max()
			ymin = real_target_box[:, 1].min()
			ymax = real_target_box[:, 1].max()

			width = xmax - xmin
			height = ymax - ymin
			_target_box = target_bbox.copy()
			_target_box[:, 0] -= xmin
			_target_box[:, 1] -= ymin
			_M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
			warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
			warped = np.array(warped, np.uint8)
			if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
				print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
					ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
				return image
			image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
												   image[ymin:ymax, xmin:xmax])
		return image

	def add_affinity_character(self, image, target_bbox):
		return self.add_region_character(image, target_bbox, self.affinitybox)

	def add_affinity(self, image, bbox_1, bbox_2):
		center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
		tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
		bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
		tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
		br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

		affinity = np.array([tl, tr, br, bl])

		return self.add_affinity_character(image, affinity.copy())

	def generate_region(self, image_size, bboxes):
		height, width = image_size[0], image_size[1]
		target = np.zeros([height, width], dtype=np.uint8)
		for i in range(len(bboxes)):
			character_bbox = np.array(bboxes[i].copy())
			for j in range(bboxes[i].shape[0]):
				target = self.add_region_character(target, character_bbox[j])

		return target

	def generate_affinity(self, image_size, bboxes):
		height, width = image_size[0], image_size[1]
		target = np.zeros([height, width], dtype=np.uint8)
		for i in range(len(bboxes)):
			character_bbox = np.array(bboxes[i])
			total_letters = 0
			for char_num in range(character_bbox.shape[0] - 1):
				target = self.add_affinity(target, character_bbox[total_letters],
													 character_bbox[total_letters + 1])
				total_letters += 1
		return target

gaussianTransformer = GaussianTransformer()

def random_crop(imgs):
	h, w = imgs[0].shape[0:2]
	crop_h = 480
	crop_w = 640
	i = random.randint(0, h - crop_h)
	j = random.randint(0, w - crop_w)

	for idx in range(len(imgs)):
		if len(imgs[idx].shape) == 3:
			imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
		else:
			imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

	return imgs

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

		image = PIL.Image.open(image_path)
		image = np.array(image)
		image = image[:,:,:3]

		with open(label_path) as f:
			label = json.loads(f.read())

		word_bboxes = []
		char_bboxes = []
		for textline in label['textlines']:
			for char in textline['chars']:
				if char['text'] == ' ':
					if len(char_bboxes):
						word_bboxes.append(np.array(char_bboxes))
						char_bboxes = []
					continue
				left = math.floor(char['bbox']['left'])
				top = math.floor(char['bbox']['top'])
				right = math.ceil(char['bbox']['right']) - 1
				bottom = math.ceil(char['bbox']['bottom']) - 1
				if left == right:
					raise Exception("Fuck")
				if top == bottom:
					raise Exception("Fuck")
				char_bboxes.append([
					[left, top],
					[right, top],
					[right, bottom],
					[left, bottom],
				])
			if len(char_bboxes):
				word_bboxes.append(np.array(char_bboxes))
				char_bboxes = []

		region_scores = gaussianTransformer.generate_region((image.shape[0], image.shape[1]), word_bboxes)
		affinity_scores = gaussianTransformer.generate_affinity((image.shape[0], image.shape[1]), word_bboxes)

		return image, region_scores, affinity_scores

	def load_data(self, idx):
		if not self.cache[idx]:
			self.cache[idx] = self._load_data(idx)
		return self.cache[idx]

	def __getitem__(self, idx):
		image, region_scores, affinity_scores = self.load_data(idx)

		image, region_scores, affinity_scores = random_crop([image, region_scores, affinity_scores])

		# vis = visdom.Visdom()
		# vis.image(np.transpose(image, [2, 0, 1]))
		# vis.image(region_scores)
		# vis.image(affinity_scores)
		# exit()

		image = normalizeMeanVariance(image)
		image = torch.from_numpy(image).float().permute(2, 0, 1)
		region_scores = torch.from_numpy(region_scores / 255).float()
		affinity_scores = torch.from_numpy(affinity_scores / 255).float()

		return image, region_scores, affinity_scores

datasets = []

for root, dirs, files in os.walk(args.dataset_folder):
	labels_count = len([file for file in files if os.path.splitext(file)[1] == '.json'])
	if not labels_count:
		continue
	datasets.append(Dataset(root, labels_count))

dataset = torch.utils.data.ConcatDataset(datasets)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

if __name__ == '__main__':
	for data in data_loader:
		print(data[0].shape, data[1].shape, data[2].shape)
		exit(0)
