
from .dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from .crnn import CRNN, alphabet
from .utils import strLabelConverter
import os

def adjust_learning_rate(optimizer, step):
	lr = 1e-3 * (0.5 ** (step // 10000))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

step_start = 0
step_finish = 100000
step = step_start

net = CRNN()
if step != 0:
	net.load_state_dict(torch.load("checkpoints/" + str(step) + ".pt"))
net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

criterion = nn.CTCLoss()

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

converter = strLabelConverter(alphabet)

print("START")

try:
	while True:
		for images, texts, images_widths in data_loader:
			label, label_size = converter.encode(texts)

			images = images.to(device)
			label = label.to(device)
			label_size = label_size.to(device)

			preds = net(images)
			ratio = images.shape[-1] / preds.shape[0]
			preds_size = (images_widths / ratio).ceil().long()

			optimizer.zero_grad()
			loss = criterion(preds, label, preds_size, label_size)
			loss /= images.size(0)
			loss.backward()
			optimizer.step()

			writer.add_scalar("loss/1-ctc", loss * 100, step)

			step += 1

			if step % 10000 == 0:
				torch.save(net.state_dict(), "checkpoints/" + str(step) + ".pt")
				adjust_learning_rate(optimizer, step)
				print("CHECKPOINT " + str(step))

			if step >= step_finish:
				print("FINISH")
				os._exit(0)

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "exception.pt")
	print("EXCEPTION")
