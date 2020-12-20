
from dataset import create_data_loader
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from model import Model
from alphabet import alphabet
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
args = parser.parse_args()

data_loader = create_data_loader(args.dataset_folder)

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

criterion = nn.CTCLoss()

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

print("START")

try:
	while True:
		for data in data_loader:
			image = data["image"].to(device)
			text = data["text"]

			label, label_size = alphabet.encode(text)
			label = label.to(device)
			label_size = label_size.to(device)

			pred = net(image)
			pred_size = torch.Tensor([pred.size(0)] * pred.size(1)).long()

			optimizer.zero_grad()
			loss = criterion(pred, label, pred_size, label_size)
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
