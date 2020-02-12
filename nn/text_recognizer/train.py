
from .dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from .model import Model
from .alphabet import alphabet
import os

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
		for images, texts, images_size in data_loader:
			label, label_size = alphabet.encode(texts)

			images = images.to(device)
			label = label.to(device)
			label_size = label_size.to(device)

			preds = net(images)
			preds_size = images_size // 4 + 1

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

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "exception.pt")
	print("EXCEPTION")
