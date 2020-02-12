
from .dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from .model import Model
import os

loss_fn = nn.MSELoss(reduction='none')

def criterion(pred, label, label_mask):
	loss = loss_fn(pred, label)

	loss = loss.view(-1)
	label = label.view(-1)
	label_mask = label_mask.view(-1)

	posi_indexes = label >= 0.5
	nega_indexes = label < 0.5

	posi_pixels = loss[posi_indexes] * label_mask[posi_indexes]
	nega_pixels = loss[nega_indexes] * (1.0 - label_mask[nega_indexes])
	if len(posi_pixels):
		if len(nega_pixels) < 3*len(posi_pixels):
			return posi_pixels.mean() + nega_pixels.mean()
		else:
			return posi_pixels.mean() + nega_pixels.topk(3*len(posi_pixels))[0].mean()
	else:
		return nega_pixels.topk(500)[0].mean()

def adjust_learning_rate(optimizer, step):
	lr = 1e-3 * (0.9 ** (step // 2000))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

step_start = 0
step = step_start

net = Model()
if step != 0:
	net.load_state_dict(torch.load("checkpoints/" + str(step) + ".pt"))
net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

print("START")

try:
	while True:
		for data in data_loader:
			image = data[0].to(device)
			label = data[1].to(device)
			label_mask = data[2].to(device)

			pred = net(image)

			optimizer.zero_grad()
			loss = criterion(pred, label, label_mask)
			writer.add_scalar("loss", loss * 100, step)
			loss.backward()
			optimizer.step()

			step += 1

			if step % 2000 == 0:
				torch.save(net.state_dict(), "checkpoints/" + str(step) + ".pt")
				adjust_learning_rate(optimizer, step)
				print("CHECKPOINT " + str(step))

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "exception.pt")
	print("EXCEPTION")
