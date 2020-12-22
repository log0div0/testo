
import os
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime
from dataset import classes_names

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True)
args = parser.parse_args()

def is_n_times_div_by_2(value, n):
	for _ in range(n):
		if value % 2 != 0:
			return False
		value /= 2
	return True

def nearest_n_times_div_by_2(value, n):
	while True:
		if is_n_times_div_by_2(value, n):
			return value
		value += 1

def pad_image(image):
	c, h, w = image.shape
	new_h = nearest_n_times_div_by_2(h, 5)
	new_w = nearest_n_times_div_by_2(w, 5)
	new_image = np.zeros([c, new_h, new_w], dtype=np.float32)
	new_image[:, :h, :w] = image
	return new_image

ort_session = onnxruntime.InferenceSession("model.onnx")

image = Image.open(args.img)
image = np.array(image)
image = image[:,:,:3]
h, w, c = image.shape

x = image / 255
x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
x = x.transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
x = pad_image(x)
x = x[np.newaxis]                # [c, h, w] to [b, c, h, w]

ort_inputs = {"input": x}
ort_outs = ort_session.run(None, ort_inputs)
y = ort_outs[0][0]

def xywh2xyxy(x):
	y = np.empty(x.shape)
	y[..., 0] = x[..., 0] - x[..., 2] / 2
	y[..., 1] = x[..., 1] - x[..., 3] / 2
	y[..., 2] = x[..., 0] + x[..., 2] / 2
	y[..., 3] = x[..., 1] + x[..., 3] / 2
	return y

def bbox_iou(box1, box2):
	b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
	b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

	inter_rect_x1 = np.maximum(b1_x1, b2_x1)
	inter_rect_y1 = np.maximum(b1_y1, b2_y1)
	inter_rect_x2 = np.minimum(b1_x2, b2_x2)
	inter_rect_y2 = np.minimum(b1_y2, b2_y2)
	inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
		inter_rect_y2 - inter_rect_y1 + 1, 0, None
	)

	b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
	b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

	iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
	return iou


def non_max_suppression(prediction, conf_thres=0.6, nms_thres=0.4):
	prediction[:, :4] = xywh2xyxy(prediction[:, :4])
	prediction = prediction[prediction[:, 4] >= conf_thres]
	if not len(prediction):
		return []
	score = prediction[:, 4] * prediction[:, 5:].max(1)
	prediction = prediction[(-score).argsort()]

	class_confs = prediction[:, 5:].max(1)
	class_preds = prediction[:, 5:].argmax(1)
	detections = np.concatenate((
		prediction[:, :5],
		class_confs[:, np.newaxis],
		class_preds[:, np.newaxis].astype(float)
	), 1)

	output = []
	while len(detections):
		large_overlap = bbox_iou(detections[0, :4][np.newaxis], detections[:, :4]) > nms_thres
		label_match = detections[0, -1] == detections[:, -1]
		invalid = large_overlap & label_match
		weights = detections[invalid, 4:5]
		detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
		output.append(detections[0])
		detections = detections[~invalid]
	return output

y = non_max_suppression(y)

image = Image.open(args.img)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 16)

colors = ['#f08080', '#6a5acd', '#20b2aa', '#db7093', '#156943', '#0BD055', '#ACD338']

for i, pred in enumerate(y):
	x1, y1, x2, y2, obj_conf, class_conf, class_indx = pred
	class_name = classes_names[int(class_indx)]
	color = colors[i % len(colors)]
	draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
	draw.rectangle(((x1, y1 - 15), (x1 + len(class_name) * 10, y1)), fill=color, width=2)
	draw.text((x1 + 3, y1 - 16), class_name, font=font, fill="white")

image.save("output.png")
