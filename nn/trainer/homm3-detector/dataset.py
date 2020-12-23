
import torch, torchvision
from torchvision import transforms
import os
import json
import numpy as np
from PIL import Image
import argparse

transformer = transforms.Compose([
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

classes_names = [
	"cancel",
	"cursed_temple",
	"garden_of_revelation",
	"hall",
	"hero",
	"hovel",
	"mage_guild",
	"magic_spring",
	"ok",
	"peasant",
	"redwood_observatory",
	"scholar",
	"sign",
	"skeleton",
	"spell_destroy_undead",
	"spell_slow",
	"star_axis",
	"town",
]

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

	def load_json(self, path):
		with open(os.path.join(self.dataset_folder, path)) as file:
		    return json.load(file)

	def __getitem__(self, idx):
		images = []
		labels = []
		example_index = 0
		for example in self.metadata['batches'][idx]['examples']:
			img = self.load_img(example["img_path"])
			C, H, W = img.shape
			images.append(img)
			meta = self.load_json(example["meta_path"])
			for obj in meta["objs"]:
				if len(obj["tag"]):
					class_index = classes_names.index(obj["tag"])
				else:
					class_index = -1
				x = obj["x"] + (obj["width"] / 2)
				y = obj["y"] + (obj["height"] / 2)
				w = obj["width"]
				h = obj["height"]
				ignore_while_training = obj["ignore_while_training"]
				labels.append(torch.FloatTensor([
					example_index,
					class_index,
					x,
					y,
					w,
					h,
					not ignore_while_training
				]))
			example_index += 1
		return {
			"images": torch.stack(images),
			"labels": torch.stack(labels)
		}

def create_dataset_loader(dataset_folder):
	dataset = Dataset(dataset_folder)
	return torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None, shuffle=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset_folder', required=True)
	args = parser.parse_args()

	data_loader = create_dataset_loader(args.dataset_folder)

	for data in data_loader:
		for key, value in data.items():
			print(key, value.shape)
		exit(0)
