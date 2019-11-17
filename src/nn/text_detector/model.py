
import dataset as generator
import torch.nn as nn

def ModelBlock(in_channels, out_channels, kernel_size):
	return nn.Sequential(
		nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
								padding=kernel_size // 2, bias=False),
		nn.BatchNorm2d(num_features=out_channels),
		nn.LeakyReLU(negative_slope=0.1),
		nn.Dropout2d(p=0.03)
	)

def Model(height = None, width = None):
	return nn.Sequential(
		ModelBlock(in_channels=3, out_channels=8, kernel_size=3),
		nn.MaxPool2d(kernel_size=2, stride=2),
		ModelBlock(in_channels=8, out_channels=32, kernel_size=3),
		nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
		ModelBlock(in_channels=32, out_channels=128, kernel_size=3),
		nn.MaxPool2d(kernel_size=2, stride=2),
		ModelBlock(in_channels=128, out_channels=256, kernel_size=3),
		ModelBlock(in_channels=256, out_channels=256, kernel_size=1),
		nn.Conv2d(in_channels=256, out_channels=1 + 2 + 2 + len(generator.symbols) + len(generator.colors) + len(generator.colors), kernel_size=1),
		nn.Sigmoid()
	)
