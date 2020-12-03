
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
	transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def create_BCELogit_loss_label(label_size, pos_thr, neg_thr):
	center = (label_size - 1) / 2
	line = np.arange(0, label_size)
	line = line - center
	line = line**2
	line = np.expand_dims(line, axis=0)
	dist_map = line + line.transpose()

	label = np.zeros([label_size, label_size, 2]).astype(np.float32)
	label[:, :, 0] = dist_map <= pos_thr**2
	label[:, :, 1] = dist_map > neg_thr**2

	return torch.from_numpy(label)

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
		search_imgs = []
		reference_imgs = []
		labels = []
		for example in self.metadata['batches'][idx]['examples']:
			search_imgs.append(self.load_img(example["search_img_path"]))
			reference_imgs.append(self.load_img(example["reference_img_path"]))
			labels.append(create_BCELogit_loss_label(49, 3, 6))
		return {
			"reference_img": torch.stack(reference_imgs),
			"search_img": torch.stack(search_imgs),
			"label": torch.stack(labels)
		}

dataset = Dataset(args.dataset_folder)

data_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, shuffle=True)

if __name__ == '__main__':
	for data in data_loader:
		for key, value in data.items():
			print(key, value.shape)
		exit(0)
