
from .dataset import data_loader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import traceback
from .craft import CRAFT
import os

class Criterion(nn.Module):
	def single_image_loss(self, pre_loss, loss_label):
		batch_size = pre_loss.shape[0]
		sum_loss = torch.mean(pre_loss.view(-1))*0
		pre_loss = pre_loss.view(batch_size, -1)
		loss_label = loss_label.view(batch_size, -1)
		internel = batch_size
		for i in range(batch_size):
			average_number = 0
			loss = torch.mean(pre_loss.view(-1)) * 0
			positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
			average_number += positive_pixel
			if positive_pixel != 0:
				posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
				sum_loss += posi_loss
				if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
					nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
					average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
				else:
					nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
					average_number += 3*positive_pixel
				sum_loss += nega_loss
			else:
				nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
				average_number += 500
				sum_loss += nega_loss
			#sum_loss += loss/average_number

		return sum_loss

	def forward(self, pred, region_scores, affinity_scores):
		gh_label = region_scores
		gah_label = affinity_scores
		p_gh = pred[:, :, :, 0]
		p_gah = pred[:, :, :, 1]

		loss_fn = torch.nn.MSELoss(reduction='none')

		assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
		loss1 = loss_fn(p_gh, gh_label)
		loss2 = loss_fn(p_gah, gah_label)

		char_loss = self.single_image_loss(loss1, gh_label)
		affi_loss = self.single_image_loss(loss2, gah_label)

		writer.add_scalar("loss/1-char", char_loss * 100, step)
		writer.add_scalar("loss/2-affi", affi_loss * 100, step)

		return char_loss/loss1.shape[0] + affi_loss/loss2.shape[0]

def adjust_learning_rate(optimizer, step):
	lr = 1e-3 * (0.5 ** (step // 10000))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

step_start = 0
step_finish = 100000
step = step_start

net = CRAFT()
if step != 0:
	net.load_state_dict(torch.load("checkpoints/" + str(step) + ".pt"))
net.to(device)
net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

criterion = Criterion()

if not os.path.isdir("checkpoints"):
	os.mkdir("checkpoints")

adjust_learning_rate(optimizer, step)

print("START")

try:
	while True:
		for data in data_loader:
			image = data[0].to(device)
			region_scores = data[1].to(device)
			affinity_scores = data[2].to(device)

			pred = net(image)

			optimizer.zero_grad()
			loss = criterion(pred, region_scores, affinity_scores)
			loss.backward()
			optimizer.step()

			step += 1

			if step >= step_finish:
				print("FINISH")
				os._exit(0)

			if step % 10000 == 0:
				torch.save(net.state_dict(), "checkpoints/" + str(step) + ".pt")
				adjust_learning_rate(optimizer, step)
				print("CHECKPOINT " + str(step))

except:
	traceback.print_exc()
	torch.save(net.state_dict(), "exception.pt")
	print("EXCEPTION")
