
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import (
	builder,
	images_count,
	colors,
	symbols
)
from model import Model

np.set_printoptions(precision=2, suppress=True)

def preprocess_image_and_label(example):
	image = example['image']
	image /= 255
	label = example['label']
	return image, label

dataset = builder.as_dataset(split='train')
dataset = dataset.shuffle(images_count)
dataset = dataset.map(preprocess_image_and_label)
dataset = dataset.batch(32)

model = Model()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

K = tf.keras.backend

def binary_focal_loss(y_true, y_pred, gamma=2.):
	pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
	epsilon = K.epsilon()
	pt = K.clip(pt, epsilon, 1. - epsilon)
	cross_entropy = -K.log(pt)
	loss = K.pow(1 - pt, gamma) * cross_entropy
	return K.sum(loss, axis=-1)

def categorical_focal_loss(y_true, y_pred, gamma=2.):
	epsilon = K.epsilon()
	y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
	cross_entropy = -y_true * K.log(y_pred)
	loss = K.pow(1 - y_pred, gamma) * cross_entropy
	return K.sum(loss, axis=-1)

run_eagerly = True

def Loss(y_true, y_pred):
	pred_obj, pred_xy, pred_wh, pred_symbol, pred_fg, pred_bg = tf.split(
		y_pred, (1, 2, 2, len(symbols), len(colors), len(colors)), axis=-1)

	true_obj, true_xy, true_wh, true_symbol, true_fg, true_bg = tf.split(
		y_true, (1, 2, 2, len(symbols), len(colors), len(colors)), axis=-1)

	obj_loss = binary_focal_loss(true_obj, pred_obj)
	obj_mask = tf.squeeze(true_obj, -1)
	xy_loss = obj_mask * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
	wh_loss = obj_mask * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
	symbol_loss = obj_mask * categorical_focal_loss(true_symbol, pred_symbol)
	fg_loss = obj_mask * categorical_focal_loss(true_fg, pred_fg)
	bg_loss = obj_mask * categorical_focal_loss(true_bg, pred_bg)

	obj_loss = tf.math.reduce_mean(obj_loss)
	xy_loss = tf.math.reduce_mean(xy_loss) * 0.25
	wh_loss = tf.math.reduce_mean(wh_loss) * 0.25
	symbol_loss = tf.math.reduce_mean(symbol_loss)
	fg_loss = tf.math.reduce_mean(fg_loss) * 0.25
	bg_loss = tf.math.reduce_mean(xy_loss) * 0.25

	total_loss = obj_loss + xy_loss + wh_loss + symbol_loss + fg_loss + bg_loss

	if run_eagerly:
		with tf.name_scope("") as scope:
			tf.summary.scalar("total_loss", total_loss, optimizer.iterations)
			tf.summary.scalar("loss/1-obj", obj_loss, optimizer.iterations)
			tf.summary.scalar("loss/2-xy", xy_loss, optimizer.iterations)
			tf.summary.scalar("loss/3-wh", wh_loss, optimizer.iterations)
			tf.summary.scalar("loss/4-symbol", symbol_loss, optimizer.iterations)
			tf.summary.scalar("loss/5-fg", fg_loss, optimizer.iterations)
			tf.summary.scalar("loss/6-bg", bg_loss, optimizer.iterations)

	return total_loss

if run_eagerly:
	summary_writer = tf.summary.create_file_writer('logs/base')
	summary_writer.set_as_default()

model.compile(optimizer=optimizer, loss=Loss, run_eagerly=run_eagerly)
callbacks = [
	tf.keras.callbacks.ModelCheckpoint('checkpoints/epoch_{epoch}.tf', verbose=1, save_weights_only=True),
]
model.fit(dataset, epochs=30, callbacks=callbacks)
model.save_weights('checkpoints/final.tf')
