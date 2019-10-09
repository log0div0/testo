
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import dataset as generator
from model import Model

np.set_printoptions(precision=2, suppress=True)

images_paths = [generator.images_dir + "/" + str(i) + ".png" for i in range(generator.images_count)]
labels_paths = [generator.labels_dir + "/" + str(i) + ".txt" for i in range(generator.images_count)]

dataset = tf.data.Dataset.from_tensor_slices((images_paths, labels_paths))

grid_w = generator.columns_count * 2
grid_h = generator.rows_count * 2

def preprocess_image_and_label(image_path, label_path):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_image(image)
	image /= 255
	image.set_shape([generator.image_height, generator.image_width, 3])
	label = tf.io.read_file(label_path)
	label = tf.strings.split(label, "\n")
	label = tf.strings.split(label[:-1])
	label = tf.strings.to_number(label)
	label = label.to_tensor()
	indexes = tf.TensorArray(tf.int32, 0, dynamic_size=True)
	updates = tf.TensorArray(tf.float32, 0, dynamic_size=True)
	for row in label:
		box_xy = row[0:2]
		grid_xy = tf.cast(box_xy * [grid_w, grid_h], tf.int32)
		indexes = indexes.write(indexes.size(), [grid_xy[1], grid_xy[0]])
		updates = updates.write(updates.size(), tf.concat([[1], row], axis=0))
	label = tf.scatter_nd(indexes.stack(), updates.stack(), [grid_h, grid_w, tf.shape(label)[-1] + 1])
	return image, label

dataset = dataset.shuffle(generator.images_count)
dataset = dataset.map(preprocess_image_and_label)
dataset = dataset.batch(32)

model = Model()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

def Loss(y_true, y_pred):
	pred_obj, pred_xy, pred_wh, pred_symbol, pred_fg, pred_bg = tf.split(
		y_pred, (1, 2, 2, len(generator.symbols), len(generator.colors), len(generator.colors)), axis=-1)

	true_obj, true_xy, true_wh, true_symbol, true_fg, true_bg = tf.split(
		y_true, (1, 2, 2, 1, 1, 1), axis=-1)

	obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)

	grid_size = tf.cast([grid_w, grid_h], tf.float32)
	grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
	grid = tf.stack(grid, axis=-1)
	grid = tf.cast(grid, tf.float32)
	true_xy = true_xy * grid_size - grid

	true_wh = true_wh * [generator.image_width, generator.image_height] / [generator.char_width, generator.char_height]

	obj_mask = tf.squeeze(true_obj, -1)
	xy_loss = obj_mask * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
	wh_loss = obj_mask * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
	symbol_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_symbol, pred_symbol)
	fg_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_fg, pred_fg)
	bg_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_bg, pred_bg)

	return obj_loss + xy_loss + wh_loss + symbol_loss + fg_loss + bg_loss

model.compile(optimizer=optimizer, loss=Loss)

callbacks = [
	tf.keras.callbacks.TensorBoard(log_dir='logs'),
	tf.keras.callbacks.ModelCheckpoint('checkpoints/epoch_{epoch}.tf', verbose=1, save_weights_only=True),
]

model.fit(dataset, epochs=20, callbacks=callbacks)
model.save_weights('checkpoints/final.tf')
