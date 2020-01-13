
import torch
import argparse
from .model import Model

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

dummy_input = torch.randn(1, 3, 480, 640)

net = Model()
net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
net.eval()

torch.onnx.export(net, dummy_input, "model.onnx",
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
	}
)

print("OK")
