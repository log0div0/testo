
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

class Yolo(nn.Module):
	def __init__(self, anchors):
		super().__init__()
		self.anchors = anchors
		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()
		self.obj_scale = 1
		self.noobj_scale = 100

	def bbox_wh_iou(self, w1, h1, w2, h2):
		inter_area = torch.min(w1, w2) * torch.min(h1, h2)
		union_area = w1 * h1 + w2 * h2 - inter_area
		return inter_area / (union_area + 1e-16)

	def forward(self, x, img_w, img_h, labels=None):
		B, C, H, W = x.shape

		prediction = x.view(B, len(self.anchors), num_classes + 5, H, W) \
			.permute(0, 1, 3, 4, 2) \
			.contiguous()

		stride_x = img_w / W
		stride_y = img_h / H

		grid_x = torch.arange(W).repeat(H, 1).view([1, 1, H, W]).to(x.device)
		grid_y = torch.arange(H).repeat(W, 1).t().view([1, 1, H, W]).to(x.device)

		anchor_w = x.new_tensor([anchor[0] for anchor in self.anchors]).view((1, len(self.anchors), 1, 1))
		anchor_h = x.new_tensor([anchor[1] for anchor in self.anchors]).view((1, len(self.anchors), 1, 1))

		pred_x = (prediction[..., 0].sigmoid() + grid_x) * stride_x
		pred_y = (prediction[..., 1].sigmoid() + grid_y) * stride_y
		pred_w = prediction[..., 2].exp() * anchor_w
		pred_h = prediction[..., 3].exp() * anchor_h
		pred_conf = prediction[..., 4].sigmoid()
		pred_cls = prediction[..., 5:].sigmoid()

		if self.train:
			lB, lCls, lX, lY, lW, lH, lIgnoreMask = labels.t()
			lB = lB.long()
			lCls = lCls.long()
			lIgnoreMask = lIgnoreMask.bool()
			gI = ((lX / img_w) * W).long()
			gJ = ((lY / img_h) * H).long()

			anchors = x.new_tensor(self.anchors)
			ious = torch.stack([self.bbox_wh_iou(anchor[0], anchor[1], lW, lH) for anchor in anchors])
			_, best_anchor = ious.max(0)

			obj_mask = x.new_zeros((B, len(self.anchors), H, W), dtype=torch.bool)
			noobj_mask = x.new_ones((B, len(self.anchors), H, W), dtype=torch.bool)
			target_x = x.new_zeros((B, len(self.anchors), H, W))
			target_y = x.new_zeros((B, len(self.anchors), H, W))
			target_w = x.new_zeros((B, len(self.anchors), H, W))
			target_h = x.new_zeros((B, len(self.anchors), H, W))
			target_cls = x.new_zeros((B, len(self.anchors), H, W, num_classes))

			obj_mask[lB, best_anchor, gJ, gI] = 1
			obj_mask[lB, best_anchor, gJ, gI] *= lIgnoreMask
			noobj_mask[lB, :, gJ, gI] = 0
			target_x[lB, best_anchor, gJ, gI] = lX
			target_y[lB, best_anchor, gJ, gI] = lY
			target_w[lB, best_anchor, gJ, gI] = lW
			target_h[lB, best_anchor, gJ, gI] = lH
			target_cls[lB, best_anchor, gJ, gI, lCls] = 1

			target_conf = obj_mask.float()

			loss_x = self.mse_loss(pred_x[obj_mask], target_x[obj_mask])
			loss_y = self.mse_loss(pred_y[obj_mask], target_y[obj_mask])
			loss_w = self.mse_loss(pred_w[obj_mask], target_w[obj_mask])
			loss_h = self.mse_loss(pred_h[obj_mask], target_h[obj_mask])
			loss_conf_obj = self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask])
			loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], target_conf[noobj_mask])
			loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
			loss_cls = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask])

			return loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
		else:
			return torch.cat(
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

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		anchors_1 = [(81,82), (135,169), (344,319)]
		anchors_2 = [(10,14), (23,27), (37,58)]
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
			Conv(512, (num_classes + 5) * len(anchors_1), 1, activation='linear'),
			Yolo(anchors_1),

			Route([-4]),
			Conv(256, 128, 1),
			Upsample(2),
			Route([-1, 8]),
			Conv(128 + 256, 256, 3),
			Conv(256, (num_classes + 5) * len(anchors_2), 1, activation='linear'),
			Yolo(anchors_2)
		])

	def forward(self, img, labels=None):
		layer_outputs = []

		if self.train:
			loss = 0
		else:
			yolo_outputs = []

		x = img
		for module in self.module_list:
			if isinstance(module, Route):
				x = torch.cat([layer_outputs[i] for i in module.layers_indexes], 1)
			elif isinstance(module, Yolo):
				x = module(x, img.shape[3], img.shape[2], labels)
				if self.train:
					loss += x
				else:
					yolo_outputs.append(x)
			else:
				x = module(x)
			layer_outputs.append(x)

		if self.train:
			return loss
		else:
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
