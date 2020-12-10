
from dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import traceback
from model import Model
import os

def focal_loss(inputs, targets, alpha=1, gamma=2):
	BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
	pt = torch.exp(-BCE_loss)
	F_loss = alpha * (1-pt)**gamma * BCE_loss
	return F_loss

def criterion(pred, label):
	label = label.unsqueeze(1)
	loss = focal_loss(pred, label[:, :, :, :, 0])
	posi_pixels = label[:, :, :, :, 0]
	nega_pixels = label[:, :, :, :, 1]
	posi_loss = loss * posi_pixels
	nega_loss = loss * nega_pixels
	return ((posi_loss.sum() / posi_pixels.sum()) + (nega_loss.sum() / nega_pixels.sum()))

def adjust_learning_rate(optimizer, step):
	if step > 20000:
		step -= 20000
	else:
		step = 0
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
			search_img = data["search_img"].to(device)
			reference_img = data["reference_img"].to(device)
			label = data["label"].to(device)

			pred = net(reference_img, search_img)

			optimizer.zero_grad()
			loss = criterion(pred, label)
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
	torch.save(net.state_dict(), "checkpoints/exception.pt")
	print("EXCEPTION")
