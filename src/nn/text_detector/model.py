
import tensorflow as tf
from tensorflow.keras.layers import (
	Conv2D,
	Input,
	MaxPool2D,
	BatchNormalization,
	LeakyReLU,
	Dropout
)
import dataset as generator

def DarknetConv(x, filters, kernel_size):
	x = Conv2D(filters=filters, kernel_size=kernel_size,
							strides=1, padding="same",
							use_bias=False)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = Dropout(0.03)(x)
	return x

def Model(height = None, width = None):
	x = inputs = Input([height, width, 3])
	x = DarknetConv(x, filters=8, kernel_size=3)
	x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
	x = DarknetConv(x, filters=32, kernel_size=3)
	x = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
	x = DarknetConv(x, filters=128, kernel_size=3)
	x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
	x = DarknetConv(x, filters=256, kernel_size=3)
	x = DarknetConv(x, filters=256, kernel_size=1)
	x = Conv2D(filters=1 + 2 + 2 + len(generator.symbols) + len(generator.colors) + len(generator.colors), kernel_size=1)(x)

	obj, xy, wh, symbol, fg, bg = tf.split(
		x, (1, 2, 2, len(generator.symbols), len(generator.colors), len(generator.colors)), axis=-1)

	obj = tf.nn.sigmoid(obj)
	xy = tf.nn.sigmoid(xy)
	wh = tf.nn.relu(wh)
	symbol = tf.nn.softmax(symbol)
	fg = tf.nn.softmax(fg)
	bg = tf.nn.softmax(bg)

	x = tf.concat([obj, xy, wh, symbol, fg, bg], axis=-1)

	return tf.keras.Model(inputs, x)
