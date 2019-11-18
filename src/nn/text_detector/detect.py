
import argparse
import matplotlib.pyplot as plt
from model import Model
import dataset as generator
import numpy as np
import torch

np.set_printoptions(precision=2, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('image', help='path to input image')
args = parser.parse_args()

image = plt.imread(args.image)
image = torch.Tensor(image)
image = image.unsqueeze(0)
image = image.permute(0, 3, 1, 2)

model = Model()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
prediction = model(image)
prediction = prediction[0]

obj, xy, wh, symbol, fg, bg = prediction.split(
	(1, 2, 2, len(generator.symbols), len(generator.colors), len(generator.colors)), dim=0)

# y = 57
# for x in range(prediction.shape[2]):
# 	if (obj[0][y][x] < 0.1):
# 		continue
# 	print(x)
# 	a = symbol[:,y,x]
# 	for n in range(len(a)):
# 		if a[n] > 0.01:
# 			print(round(float(a[n]), 2), generator.symbols[n])
# exit()

symbol_index = symbol.argmax(dim=0)

for y in range(prediction.shape[1]):
	for x in range(prediction.shape[2]):
		if (obj[0][y][x] > 0.1):
			symbol = generator.symbols[symbol_index[y][x]]
			print(symbol[0], end='')
		else:
			print(' ', end='')
	print()
