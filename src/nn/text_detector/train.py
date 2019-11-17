
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import (
	images_count,
	colors,
	symbols
)
from model import Model
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=2, suppress=True)

dataset = np.load("dataset.npz")
images = torch.Tensor(dataset['images'] / 255)
images = images.permute(0, 3, 1, 2)
labels = torch.Tensor(dataset['labels'])
labels = labels.permute(0, 3, 1, 2)
dataset = torch.utils.data.TensorDataset(images, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = Model()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()

def binary_focal_loss(y_true, y_pred, gamma=2.):
	pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
	epsilon = 0.0000001
	pt = pt.clamp(epsilon, 1. - epsilon)
	cross_entropy = -pt.log()
	loss = (1 - pt).pow(gamma) * cross_entropy
	return loss.max(dim=1)[0]

step = 0

def criterion(y_true, y_pred):
	pred_obj, pred_xy, pred_wh, pred_symbol, pred_fg, pred_bg = y_pred.split(
		(1, 2, 2, len(symbols), len(colors), len(colors)), dim=1)

	true_obj, true_xy, true_wh, true_symbol, true_fg, true_bg = y_true.split(
		(1, 2, 2, len(symbols), len(colors), len(colors)), dim=1)

	obj_loss = binary_focal_loss(true_obj, pred_obj)
	obj_mask = true_obj.squeeze(dim=1)
	xy_loss = obj_mask * (true_xy - pred_xy).pow(2).max(dim=1)[0]
	wh_loss = obj_mask * (true_wh - pred_wh).pow(2).max(dim=1)[0]
	symbol_loss = obj_mask * binary_focal_loss(true_symbol, pred_symbol)
	fg_loss = obj_mask * binary_focal_loss(true_fg, pred_fg)
	bg_loss = obj_mask * binary_focal_loss(true_bg, pred_bg)

	obj_loss = obj_loss.mean() * 100
	xy_loss = xy_loss.mean() * 100 * 0.25
	wh_loss = wh_loss.mean() * 100 * 0.25
	symbol_loss = symbol_loss.mean() * 100
	fg_loss = fg_loss.mean() * 100 * 0.25
	bg_loss = xy_loss.mean() * 100 * 0.25

	total_loss = obj_loss + xy_loss + wh_loss + symbol_loss + fg_loss + bg_loss

	writer.add_scalar("total_loss", total_loss, step)
	writer.add_scalar("loss/1-obj", obj_loss, step)
	writer.add_scalar("loss/2-xy", xy_loss, step)
	writer.add_scalar("loss/3-wh", wh_loss, step)
	writer.add_scalar("loss/4-symbol", symbol_loss, step)
	writer.add_scalar("loss/5-fg", fg_loss, step)
	writer.add_scalar("loss/6-bg", bg_loss, step)

	return total_loss

for epoch in range(50):
	print("Epoch", epoch)
	for inputs, labels in dataloader:
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(labels, outputs)
		loss.backward()
		optimizer.step()
		step += 1

torch.save(model.state_dict(), "model.pt")
print("OK")
