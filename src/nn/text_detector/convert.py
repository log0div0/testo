
import torch
from model import Model

dummy_input = torch.randn(1, 3, 224, 224)

model = Model()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

torch.onnx.export(model, dummy_input, "model.onnx",
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
