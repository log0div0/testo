

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_


def Conv(in_ch, out_ch, kernel_size):
	return nn.Sequential(
		nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
		nn.BatchNorm2d(out_ch),
		nn.ReLU(inplace=True),
	)


def EmbeddingNet():
	return nn.Sequential(
		Conv(3, 64, kernel_size=3),
		Conv(64, 64, kernel_size=3),
		nn.MaxPool2d(2, stride=2),
		Conv(64, 128, kernel_size=3),
		Conv(128, 128, kernel_size=3),
		nn.MaxPool2d(2, stride=2),
		Conv(128, 256, kernel_size=3),
		Conv(256, 256, kernel_size=3),
		Conv(256, 256, kernel_size=3)
	)


class SiameseNet(nn.Module):
	def __init__(self, embedding_net):
		super().__init__()
		self.embedding_net = embedding_net
		num_class = 1
		self.conv_cls = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
			nn.Conv2d(64, num_class, kernel_size=1), nn.BatchNorm2d(1)
		)

	def forward(self, embed_ref, srch_img):
		embed_srch = self.embedding_net(srch_img)
		y = self.conv2d_dw_group(embed_srch, embed_ref)
		y = self.conv_cls(y)
		if not self.training:
			y = y.sigmoid()
		return y

	def conv2d_dw_group(self, x, kernel):
		batch, channel = kernel.shape[:2]
		x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
		kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
		out = F.conv2d(x, kernel, groups=batch*channel)
		out = out.view(batch, channel, out.size(2), out.size(3))
		return out


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.ref_embedding_net = EmbeddingNet()
		self.siamese_net = SiameseNet(EmbeddingNet())
		self.weights_init()

	def weights_init(self):
		for model in self.modules():
			if isinstance(model, nn.Conv2d):
				xavier_uniform_(model.weight, gain=math.sqrt(2.0))
				constant_(model.bias, 0.1)
			elif isinstance(model, nn.BatchNorm2d):
				normal_(model.weight, 1.0, 0.02)
				zeros_(model.bias)

	def forward(self, ref_img, srch_img):
		embed_ref = self.ref_embedding_net(ref_img)
		return self.siamese_net(embed_ref, srch_img)


if __name__ == '__main__':
	import time
	import onnxruntime
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--model')
	args = parser.parse_args()

	model = Model()
	if args.model:
		model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
	model.eval()

	print(model)
	ref_img = torch.randn(1, 3, 63, 63)
	srch_img = torch.randn(1, 3, 255, 255)
	start_time = time.time()
	output = model(ref_img, srch_img)
	print("--- %s seconds ---" % (time.time() - start_time))
	print(output.shape)

	torch.onnx.export(model, (ref_img, srch_img), "model.onnx",
		input_names=["ref_img", "srch_img"],
		output_names=["output"],
		dynamic_axes={
			'ref_img': {
				2: 'height',
				3: 'width'
			},
			'srch_img': {
				2: 'height',
				3: 'width'
			},
			'output': {
				2: 'height',
				3: 'width'
			}
		}
	)

	start_time = time.time()
	ort_session = onnxruntime.InferenceSession("model.onnx")
	ort_inputs = {
		"ref_img": ref_img.numpy(),
		"srch_img": srch_img.numpy()
	}
	ort_outs = ort_session.run(None, ort_inputs)
	y = torch.Tensor(ort_outs[0])
	print("--- %s seconds ---" % (time.time() - start_time))
	print(y.shape)
