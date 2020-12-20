
from dataset import create_dataset_loader
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from model import Model, encode_colors, colors
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
args = parser.parse_args()

data_loader = create_dataset_loader(args.dataset_folder)

def adjust_learning_rate(optimizer, step):
	lr = 1e-3 * (0.9 ** (step // 10000))
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

def criterion(pred, target):
	target = target.repeat(pred.shape[0], 1)

	pred = pred.reshape([-1, len(colors)])
	target = target.reshape([-1])

	return nn.functional.nll_loss(pred, target)

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

print("START")

try:
	while True:
		for data in data_loader:
			image = data["image"].to(device)
			fg = encode_colors(data["fg"]).to(device)
			bg = encode_colors(data["bg"]).to(device)

			optimizer.zero_grad()

			pred = net(image)
			loss_fg = criterion(pred[:,:,:len(colors)], fg)
			loss_bg = criterion(pred[:,:,len(colors):], bg)
			loss = loss_fg + loss_bg
			loss.backward()

			optimizer.step()

			writer.add_scalar("loss", loss * 100, step)

			step += 1

			if step % 10000 == 0:
				torch.save(net.state_dict(), "checkpoints/" + str(step) + ".pt")
				adjust_learning_rate(optimizer, step)
				print("CHECKPOINT " + str(step))

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "checkpoints/exception.pt")
	print("EXCEPTION")
