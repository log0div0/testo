
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .alphabet import alphabet

MAX_HEIGHT = 32

class BidirectionalLSTM(nn.Module):
	def __init__(self, nIn, nHidden, nOut):
		super(BidirectionalLSTM, self).__init__()

		self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
		self.embedding = nn.Linear(nHidden * 2, nOut)

	def forward(self, input):
		recurrent, _ = self.rnn(input)
		T, b, h = recurrent.size()
		t_rec = recurrent.view(T * b, h)

		output = self.embedding(t_rec)  # [T * b, nOut]
		output = output.view(T, b, -1)

		return output

def ConvRelu(nIn, nOut, ks, ss, ps, batchNormalization=False):
	seq = nn.Sequential()
	seq.add_module('conv', nn.Conv2d(nIn, nOut, ks, ss, ps))
	if batchNormalization:
		seq.add_module('batchnorm', nn.BatchNorm2d(nOut))
	seq.add_module('relu', nn.LeakyReLU(0.2, inplace=True))
	return seq

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.cnn = nn.Sequential(
			ConvRelu(3, 16, 3, 1, 1),
			nn.MaxPool2d(2, 2),                    # 64x16x64
			ConvRelu(16, 32, 3, 1, 1),
			nn.MaxPool2d(2, 2),                    # 128x8x32
			ConvRelu(32, 64, 3, 1, 1, True),
			ConvRelu(64, 64, 3, 1, 1),
			nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x16
			ConvRelu(64, 128, 3, 1, 1, True),
			ConvRelu(128, 128, 3, 1, 1),
			nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x16
			ConvRelu(128, 128, 2, 1, 0, True)      # 512x1x16
		)

		nh=128
		self.rnn = nn.Sequential(
			BidirectionalLSTM(128, nh, nh),
			BidirectionalLSTM(nh, nh, len(alphabet.char_groups) + 1))

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

	def forward(self, input):
		# conv features
		conv = self.cnn(input)
		conv = conv.squeeze(2)
		conv = conv.permute(2, 0, 1)  # [w, b, c]

		# rnn features
		output = self.rnn(conv)

		# add log_softmax to converge output
		output = F.log_softmax(output, dim=2)

		return output


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
	x = torch.randn(1, 3, 32, 100)
	start_time = time.time()
	output = model(x)
	print("--- %s seconds ---" % (time.time() - start_time))
	print(output.shape)

	torch.onnx.export(model, x, "model.onnx",
		input_names=["input"],
		output_names=["output"],
			dynamic_axes={
			'input': {
				3: 'width'
			},
			'output': {
				0: 'width',
			}
		}
	)

	start_time = time.time()
	ort_session = onnxruntime.InferenceSession("model.onnx")
	ort_inputs = {"input": x.numpy()}
	ort_outs = ort_session.run(None, ort_inputs)
	y = torch.Tensor(ort_outs[0])
	print("--- %s seconds ---" % (time.time() - start_time))
	print(y.shape)
