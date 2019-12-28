
import argparse
import numpy as np
import torch
import PIL.Image

parser = argparse.ArgumentParser()
parser.add_argument('input_img_path')
parser.add_argument('--model', default='model.pt')
args = parser.parse_args()

from .crnn import CRNN, alphabet, MAX_HEIGHT
from .utils import strLabelConverter

converter = strLabelConverter(alphabet)

net = CRNN()
net.load_state_dict(torch.load(args.model, map_location='cpu'))
net.eval()

image = PIL.Image.open(args.input_img_path)
image = image.convert("L")
ratio = image.width / image.height
height = MAX_HEIGHT
width = int(ratio * height)
image = image.resize([width, height], PIL.Image.BILINEAR)

image = np.array(image, np.float32)
image /= 255.0
image -= 0.5

x = torch.from_numpy(image).unsqueeze(0)
x = x.unsqueeze(0)

preds = net(x)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.LongTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
