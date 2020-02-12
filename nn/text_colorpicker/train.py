
from .dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from .model import Model, colors
import os

torch.set_printoptions(precision=2, sci_mode=False)

def adjust_learning_rate(optimizer, step):
	lr = 1e-3 * (0.5 ** (step // 10000))
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

criterion = nn.NLLLoss(reduction='none')

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

print("START")

try:
	while True:
		for data in data_loader:
			img = data['img'].to(device)
			color = data['color'].to(device)
			background_color = data['background_color'].to(device)

			preds = net(img)

			optimizer.zero_grad()
			loss_color = criterion(preds[:,:len(colors)], color)
			loss_background_color = criterion(preds[:,len(colors):], background_color)
			loss_total = loss_color + loss_background_color
			loss_total = loss_total.topk(loss_total.size(0) // 2)[0].mean()
			loss_total.backward()
			optimizer.step()

			writer.add_scalar("loss/0-total", loss_total * 100, step)

			step += 1

			if step % 10000 == 0:
				torch.save(net.state_dict(), "checkpoints/" + str(step) + ".pt")
				adjust_learning_rate(optimizer, step)
				print("CHECKPOINT " + str(step))

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "exception.pt")
	print("EXCEPTION")
