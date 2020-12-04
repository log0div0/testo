
import os
import math
import argparse
import numpy as np
from PIL import Image
import onnxruntime
from model import colors

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True)
args = parser.parse_args()

ort_session = onnxruntime.InferenceSession("model.onnx")

image = Image.open(args.img)
ratio = math.ceil(image.width / image.height)
image = image.resize([ratio * 32 * 2, 32])
image.save("output.png")
image = np.array(image)
image = image[:,:,:3]
h, w, c = image.shape

x = image / 255
x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
x = x.transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
x = x[np.newaxis]                # [c, h, w] to [b, c, h, w]
x = x.astype(np.float32)

ort_inputs = {"input": x}
ort_outs = ort_session.run(None, ort_inputs)
y = ort_outs[0]

def print_pred(pred):
	stat = {}
	for item in pred:
		item = item.squeeze()
		indexes = range(item.shape[0])
		indexes = sorted(indexes, key=lambda i: -item[i])
		color = colors[indexes[0]]
		if color in stat:
			stat[color] += 1
		else:
			stat[color] = 0
	for color in stat:
		print(color, stat[color])

print("================ fg ================")
print_pred(y[:,:,:len(colors)])
print("================ bg ================")
print_pred(y[:,:,len(colors):])