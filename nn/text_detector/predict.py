
import os
import math
import argparse
import numpy as np
import torch
import PIL.Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('input_folder')
parser.add_argument('--model', default='model.pt')
parser.add_argument('--output_folder', default='output')
args = parser.parse_args()

def list_files(in_path):
	img_files = []
	for (dirpath, dirnames, filenames) in os.walk(in_path):
		for file in filenames:
			filename, ext = os.path.splitext(file)
			if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
				img_files.append(os.path.join(dirpath, file))
	return img_files

def loadImage(img_file):
	image = PIL.Image.open(img_file)
	image = np.array(image)
	if image.shape[2] == 4:   image = image[:,:,:3]
	return image

def cvt2HeatmapImg(img):
	img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
	img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
	return img

""" auxilary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
	out = np.matmul(Minv, (pt[0], pt[1], 1))
	return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
	# prepare data
	linkmap = linkmap.copy()
	textmap = textmap.copy()
	img_h, img_w = textmap.shape

	""" labeling method """
	ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
	ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

	text_score_comb = np.clip(text_score + link_score, 0, 1)
	nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

	det = []
	for k in range(1,nLabels):
		# size filtering
		size = stats[k, cv2.CC_STAT_AREA]
		if size < 10: continue

		# thresholding
		if np.max(textmap[labels==k]) < text_threshold: continue

		# make segmentation map
		segmap = np.zeros(textmap.shape, dtype=np.uint8)
		segmap[labels==k] = 255
		segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
		x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
		w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
		niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
		sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
		# boundary check
		if sx < 0 : sx = 0
		if sy < 0 : sy = 0
		if ex >= img_w: ex = img_w
		if ey >= img_h: ey = img_h
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
		segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

		# make box
		np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
		rectangle = cv2.minAreaRect(np_contours)
		box = cv2.boxPoints(rectangle)

		# align diamond-shape
		w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
		box_ratio = max(w, h) / (min(w, h) + 1e-5)
		if abs(1 - box_ratio) <= 0.1:
			l, r = min(np_contours[:,0]), max(np_contours[:,0])
			t, b = min(np_contours[:,1]), max(np_contours[:,1])
			box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

		# make clock-wise order
		startidx = box.sum(axis=1).argmin()
		box = np.roll(box, 4-startidx, 0)
		box = np.array(box)

		det.append(box)

	return det

def saveResult(img_file, img, boxes, dirname):
	""" save text detection result one by one
	Args:
		img_file (str): image file name
		img (array): raw image context
		boxes (array): array of result file
			Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
	Return:
		None
	"""
	img = np.array(img)

	# make result file list
	filename, file_ext = os.path.splitext(os.path.basename(img_file))

	# result directory
	res_img_file = os.path.join(dirname, filename + '.png')

	if not os.path.isdir(dirname):
		os.mkdir(dirname)

	for i, box in enumerate(boxes):
		poly = np.array(box).astype(np.int32).reshape((-1))
		poly = poly.reshape(-1, 2)
		cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

	# Save result image
	cv2.imwrite(res_img_file, img)


from .craft import CRAFT, normalizeMeanVariance, pad_image

net = CRAFT()
net.load_state_dict(torch.load(args.model, map_location='cpu'))
net.eval()

for image_path in list_files(args.input_folder):
	print(image_path)
	image = loadImage(image_path)
	h, w, c = image.shape

	x = normalizeMeanVariance(image)
	x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
	x = pad_image(x)
	x = x.unsqueeze(0)                # [c, h, w] to [b, c, h, w]

	# forward pass
	y = net(x)

	y = y.squeeze(0)

	score_text = y[:h,:w,0].cpu().data.numpy()
	score_link = y[:h,:w,1].cpu().data.numpy()

	# Post-processing
	boxes = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4)

	render_img = score_text.copy()
	render_img = np.hstack((render_img, score_link))
	ret_score_text = cvt2HeatmapImg(render_img)

	saveResult(image_path, image[:,:,::-1], boxes, dirname=args.output_folder)

	# save score text
	filename, file_ext = os.path.splitext(os.path.basename(image_path))
	mask_file = os.path.join(args.output_folder, filename + '_mask.png')
	cv2.imwrite(mask_file, ret_score_text)

	del x
	del y

print("OK")
