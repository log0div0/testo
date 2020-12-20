
import torch, torchvision
from torchvision import transforms
import os
import json
import numpy as np
from PIL import Image
import argparse

transformer = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class Dataset:
	def __init__(self, dataset_folder):
		self.dataset_folder = dataset_folder
		metadata_path = os.path.join(dataset_folder, "meta.json")
		with open(metadata_path) as file:
			self.metadata = json.load(file)

	def __len__(self):
		return len(self.metadata['batches'])

	def load_img(self, path):
		img = Image.open(os.path.join(self.dataset_folder, path))
		return transformer(img)

	def __getitem__(self, idx):
		images = []
		fg = []
		bg = []
		for example in self.metadata['batches'][idx]['examples']:
			images.append(self.load_img(example["img_path"]))
			fg.append(example["fg"])
			bg.append(example["bg"])
		return {
			"image": torch.stack(images),
			"fg": fg,
			"bg": bg,
		}

def create_dataset_loader(dataset_folder):
	dataset = Dataset(dataset_folder)
	return torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, shuffle=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_folder', required=True)
	args = parser.parse_args()

	data_loader = create_data_loader(args.dataset_folder)

	for data in data_loader:
		for key, value in data.items():
			if torch.is_tensor(value):
				print(key, value.shape)
			else:
				print(key, value)
		exit(0)
