
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import matplotlib.pyplot as plt
from model import Model
import dataset as generator
import numpy as np

np.set_printoptions(precision=2, suppress=True)

flags.DEFINE_string('image', './data/girl.png', 'path to input image')

def main(_argv):
	image = tf.io.read_file(FLAGS.image)
	image = tf.image.decode_image(image)
	image /= 255
	image = tf.expand_dims(image, 0)

	model = Model()
	model.load_weights('checkpoints/final.tf')
	prediction = model(image)

	prediction = prediction[0]

	obj, xy, wh, symbol, fg, bg = tf.split(
		prediction, (1, 2, 2, len(generator.symbols), len(generator.colors), len(generator.colors)), axis=-1)

	y = 37
	for x in range(tf.shape(prediction)[1]):
		if (obj[y][x][0] < 0.1):
			continue
		print(x)
		a = symbol[y][x]
		for n in range(len(a)):
			if a[n] > 0.1:
				print(round(float(a[n]), 2), generator.symbols[n])

	symbol_index = tf.math.argmax(symbol, axis=-1)

	for y in range(tf.shape(prediction)[0]):
		for x in range(tf.shape(prediction)[1]):
			if (obj[y][x][0] > 0.1):
				symbol = generator.symbols[symbol_index[y][x]]
				print(symbol[0], end='')
			else:
				print(' ', end='')
		print()

app.run(main)
