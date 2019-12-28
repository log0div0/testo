
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def Conv(in_ch, out_ch, kernel_size):
	return nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
	)

def DoubleConv(in_ch, mid_ch, out_ch):
	return nn.Sequential(
		Conv(in_ch + mid_ch, mid_ch, kernel_size=1),
		Conv(mid_ch, out_ch, kernel_size=3)
	)

class CRAFT(nn.Module):
	def __init__(self):
		super().__init__()

		self.downconv1 = nn.Sequential(
			Conv(3, 16, kernel_size=3),
			Conv(16, 16, kernel_size=3),
			Conv(16, 32, kernel_size=3),
			Conv(32, 32, kernel_size=3)
		)
		self.downconv2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			Conv(32, 64, kernel_size=3),
			Conv(64, 64, kernel_size=3)
		)
		self.downconv3 = nn.Sequential(
			Conv(64, 64, kernel_size=3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			Conv(64, 128, kernel_size=3),
			Conv(128, 128, kernel_size=3)
		)
		self.downconv4 = nn.Sequential(
			Conv(128, 128, kernel_size=3),
			nn.MaxPool2d(kernel_size=2, stride=2),
			Conv(128, 128, kernel_size=3),
			Conv(128, 128, kernel_size=3)
		)

		self.downconv5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(128, 256, kernel_size=3, padding=6, dilation=6),
			nn.Conv2d(256, 256, kernel_size=1)
		)

		self.upconv1 = DoubleConv(256, 128, 64)
		self.upconv2 = DoubleConv(128, 64, 32)
		self.upconv3 = DoubleConv(64, 32, 16)
		self.upconv4 = DoubleConv(32, 16, 8)

		num_class = 2
		self.conv_cls = nn.Sequential(
			nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(8, 4, kernel_size=3, padding=1), nn.ReLU(inplace=True),
			nn.Conv2d(4, 4, kernel_size=1), nn.ReLU(inplace=True),
			nn.Conv2d(4, num_class, kernel_size=1),
		)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, x):
		h = h4 = self.downconv1(x)
		h = h3 = self.downconv2(h)
		h = h2 = self.downconv3(h)
		h = h1 = self.downconv4(h)
		h = h0 = self.downconv5(h)

		y = torch.cat([h0, h1], dim=1)
		y = self.upconv1(y)

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
		y = torch.cat([y, h2], dim=1)
		y = self.upconv2(y)

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
		y = torch.cat([y, h3], dim=1)
		y = self.upconv3(y)

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
		y = torch.cat([y, h4], dim=1)
		feature = self.upconv4(y)

		y = self.conv_cls(feature)

		return y.permute(0,2,3,1)

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
	# should be RGB order
	img = in_img.astype(np.float32)

	img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
	img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
	return img

def is_n_times_div_by_2(value, n=4):
	for _ in range(n):
		if value % 2 != 0:
			return False
		value /= 2
	return True

def nearest_n_times_div_by_2(value, n=4):
	while True:
		if is_n_times_div_by_2(value, n):
			return value
		value += 1

def pad_image(image):
	c, h, w = image.shape
	new_h = nearest_n_times_div_by_2(h)
	new_w = nearest_n_times_div_by_2(w)
	new_image = torch.zeros([c, new_h, new_w])
	new_image[:, :h, :w] = image
	return new_image

if __name__ == '__main__':
	import time
	import onnxruntime

	model = CRAFT()
	print(model)
	x = torch.randn(1, 3, nearest_n_times_div_by_2(600), nearest_n_times_div_by_2(800))
	start_time = time.time()
	output = model(x)
	print("--- %s seconds ---" % (time.time() - start_time))
	print(output.shape)

	torch.onnx.export(model, x, "model.onnx",
		input_names=["input"],
		output_names=["output"]
	)

	start_time = time.time()
	ort_session = onnxruntime.InferenceSession("model.onnx")
	ort_inputs = {"input": x.numpy()}
	ort_outs = ort_session.run(None, ort_inputs)
	y = torch.Tensor(ort_outs[0])
	print("--- %s seconds ---" % (time.time() - start_time))
	print(y.shape)
