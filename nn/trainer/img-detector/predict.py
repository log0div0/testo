
import os
import math
import argparse
import numpy as np
from PIL import Image
import onnxruntime
import time


parser = argparse.ArgumentParser()
parser.add_argument('--srch_img', required=True)
parser.add_argument('--ref_img', required=True)
args = parser.parse_args()


def to_tensor(img):
	img = np.array(img)
	img = img[:,:,:3]
	img = img / 255.0
	img = (img - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
	img = img.transpose([2, 0, 1])    # [h, w, c] to [c, h, w]
	img = img[np.newaxis]             # [c, h, w] to [b, c, h, w]
	return np.float32(img)


srch_img = to_tensor(Image.open(args.srch_img))
ref_img = Image.open(args.ref_img)
ref_img = ref_img.resize((63, 63))
ref_img = to_tensor(ref_img)


srch_img = np.pad(srch_img, ((0,0),(0,0),(64,64),(64,64)))
print(srch_img.shape)
print(ref_img.shape)

ort_session = onnxruntime.InferenceSession("model.onnx")
ort_inputs = {
	"ref_img": ref_img,
	"srch_img": srch_img
}
start_time = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print("--- %s seconds ---" % (time.time() - start_time))
y = ort_outs[0][0]

y = np.squeeze(y)
r = np.array(y, copy=True)
r[r > 0.9] = 0
y = np.stack([y, r, r], axis=2)
y = np.uint8(y * 255)

print(y.shape)

Image.fromarray(y).save("output.png")

print("OK")
