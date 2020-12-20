
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def Conv(in_ch, out_ch, kernel_size, activation='leaky'):
	seq = nn.Sequential(
		nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
		nn.BatchNorm2d(out_ch),
	)
	if activation == 'leaky':
		seq.add_module("2", nn.LeakyReLU(0.1))
	elif activation == 'linear':
		pass
	else:
		raise "Unknown activation"
	return seq

def MaxPool(kernel_size, stride):
	if kernel_size == 2 and stride == 1:
		return nn.Sequential(
			nn.ZeroPad2d((0, 1, 0, 1)),
			nn.MaxPool2d(kernel_size, stride)
		)
	else:
		return nn.MaxPool2d(kernel_size, stride)

class Route(nn.Module):
	def __init__(self, layers_indexes):
		super().__init__()
		self.layers_indexes = layers_indexes

class Upsample(nn.Module):
	def __init__(self, scale_factor, mode="nearest"):
		super().__init__()
		self.scale_factor = scale_factor
		self.mode = mode

	def forward(self, x):
		x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
		return x

num_classes = 2
num_anchors = 3

class Yolo(nn.Module):
	def __init__(self, anchors):
		super().__init__()
		self.anchors = anchors

	def forward(self, x, img_w, img_h):
		B, C, H, W = x.shape

		prediction = x.view(B, num_anchors, num_classes + 5, H, W) \
			.permute(0, 1, 3, 4, 2) \
			.contiguous()

		stride_x = img_w / W
		stride_y = img_h / H

		grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W])
		grid_y = torch.arange(H).repeat(W, 1).t().view([1, 1, H, W])

		anchor_w = torch.FloatTensor([anchor[0] for anchor in self.anchors]).view((1, num_anchors, 1, 1))
		anchor_h = torch.FloatTensor([anchor[1] for anchor in self.anchors]).view((1, num_anchors, 1, 1))

		pred_x = (prediction[..., 0].sigmoid() + grid_x) * stride_x
		pred_y = (prediction[..., 1].sigmoid() + grid_y) * stride_y
		pred_w = prediction[..., 2].exp() * anchor_w
		pred_h = prediction[..., 3].exp() * anchor_h
		pred_conf = prediction[..., 4].sigmoid()
		pred_cls = prediction[..., 5:].sigmoid()

		output = torch.cat(
			(
				pred_x.view(B, -1, 1),
				pred_y.view(B, -1, 1),
				pred_w.view(B, -1, 1),
				pred_h.view(B, -1, 1),
				pred_conf.view(B, -1, 1),
				pred_cls.view(B, -1, num_classes),
			),
			-1,
		)

		return output

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.module_list = nn.ModuleList([
			Conv(3, 16, 3),
			MaxPool(2, 2),
			Conv(16, 32, 3),
			MaxPool(2, 2),
			Conv(32, 64, 3),
			MaxPool(2, 2),
			Conv(64, 128, 3),
			MaxPool(2, 2),
			Conv(128, 256, 3),
			MaxPool(2, 2),
			Conv(256, 512, 3),
			MaxPool(2, 1),
			Conv(512, 1024, 3),
			#############
			Conv(1024, 256, 1),

			Conv(256, 512, 3),
			Conv(512, (num_classes + 5) * num_anchors, 1, activation='linear'),
			Yolo([(81,82), (135,169), (344,319)]),

			Route([-4]),
			Conv(256, 128, 1),
			Upsample(2),
			Route([-1, 8]),
			Conv(128 + 256, 256, 3),
			Conv(256, (num_classes + 5) * num_anchors, 1, activation='linear'),
			Yolo([(10,14), (23,27), (37,58)])
		])

	def forward(self, img):
		layer_outputs = []
		yolo_outputs = []
		x = img
		for module in self.module_list:
			if isinstance(module, Route):
				x = torch.cat([layer_outputs[i] for i in module.layers_indexes], 1)
			elif isinstance(module, Yolo):
				x = module(x, img.shape[3], img.shape[2])
				yolo_outputs.append(x)
			else:
				x = module(x)
			layer_outputs.append(x)
		return torch.cat(yolo_outputs, 1)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


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
	x = torch.randn(1, 3, 480, 640)
	start_time = time.time()
	output = model(x)
	print("--- %s seconds ---" % (time.time() - start_time))
	print(output.shape)

	torch.onnx.export(model, x, "model.onnx",
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={
			'input': {
				2: 'height',
				3: 'width'
			},
			'output': {
				2: 'height',
				3: 'width'
			}
		},
		opset_version=11
	)

	start_time = time.time()
	ort_session = onnxruntime.InferenceSession("model.onnx")
	ort_inputs = {"input": x.numpy()}
	ort_outs = ort_session.run(None, ort_inputs)
	y = torch.Tensor(ort_outs[0])
	print("--- %s seconds ---" % (time.time() - start_time))
	print(y.shape)
