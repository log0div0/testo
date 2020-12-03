
import torch, torchvision
from torchvision import transforms
import os
import json
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
args = parser.parse_args()

transformer = transforms.Compose([
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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

	def load_label(self, path):
		img = Image.open(os.path.join(self.dataset_folder, path))
		img = np.array(img)
		img = img / 255.0
		img = np.float32(img)
		img = torch.from_numpy(img)
		return img

	def __getitem__(self, idx):
		images = []
		up_labels = []
		down_labels = []
		for example in self.metadata['batches'][idx]['examples']:
			images.append(self.load_img(example["img_path"]))
			posi_up = self.load_label(example["posi_up_path"])
			posi_down = self.load_label(example["posi_down_path"])
			nega_up = self.load_label(example["nega_up_path"])
			nega_down = self.load_label(example["nega_down_path"])
			up_labels.append(torch.stack([posi_up, nega_up]))
			down_labels.append(torch.stack([posi_down, nega_down]))
		return {
			"images": torch.stack(images),
			"up_labels": torch.stack(up_labels),
			"down_labels": torch.stack(down_labels)
		}

dataset = Dataset(args.dataset_folder)

data_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, shuffle=True)

if __name__ == '__main__':
	for data in data_loader:
		for key, value in data.items():
			print(key, value.shape)
		exit(0)
