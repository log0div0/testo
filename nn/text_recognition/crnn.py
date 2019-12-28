
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

MAX_HEIGHT = 32

common = '''0123456789!?"'#$%&@()[]{}<>+-*/\\.,:;^~=|_'''
english = 'abcdefghijklmnopqrstuvwxyz'
ENGLISH = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
russian = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
RUSSIAN = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
alphabet = common + english + ENGLISH + russian + RUSSIAN

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


class CRNN(nn.Module):
	def __init__(self, nh=256, leakyRelu=False):
		super(CRNN, self).__init__()

		ks = [3, 3, 3, 3, 3, 3, 2]
		ps = [1, 1, 1, 1, 1, 1, 0]
		ss = [1, 1, 1, 1, 1, 1, 1]
		nm = [64, 128, 256, 256, 512, 512, 512]

		cnn = nn.Sequential()

		def convRelu(i, batchNormalization=False):
			nIn = 1 if i == 0 else nm[i - 1]
			nOut = nm[i]
			cnn.add_module('conv{0}'.format(i),
						   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
			if batchNormalization:
				cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
			if leakyRelu:
				cnn.add_module('relu{0}'.format(i),
							   nn.LeakyReLU(0.2, inplace=True))
			else:
				cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

		convRelu(0)
		cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
		convRelu(1)
		cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
		convRelu(2, True)
		convRelu(3)
		cnn.add_module('pooling{0}'.format(2),
					   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
		convRelu(4, True)
		convRelu(5)
		cnn.add_module('pooling{0}'.format(3),
					   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
		convRelu(6, True)  # 512x1x16

		self.cnn = cnn
		self.rnn = nn.Sequential(
			BidirectionalLSTM(512, nh, nh),
			BidirectionalLSTM(nh, nh, len(alphabet) + 1))

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

	model = CRNN()
	print(model)
	x = torch.randn(50, 1, 32, 100)
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
