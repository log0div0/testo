#!/usr/bin/python3

import os, random, colorsys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from psf import PSF

font_names = [
	'Uni2-Continent3.psf.gz',
	'Uni2-Continent4.psf.gz',
	'Uni2-Continent4-FirstScreen.psf.gz',
	'Uni2-Fixed16.psf.gz',
	'Uni2-VGA16.psf.gz'
]

colors = [
	{
		"name": "white",
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.9, 1.]
	},
	{
		"name": "gray",
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.4, .6]
	},
	{
		"name": "black",
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.0, .1]
	},
	{
		"name": "red",
		"h": [350, 370],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "orange",
		"h": [30, 36],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "yellow",
		"h": [52, 64],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "green",
		"h": [97, 125],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "cyan",
		"h": [173, 182],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "blue",
		"h": [220, 250],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	{
		"name": "purple",
		"h": [264, 281],
		"s": [.8, 1.],
		"v": [.8, 1.]
	}
]

fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
fonts = [PSF(os.path.join(fonts_dir, font_name)) for font_name in font_names]
symbols_to_glyphs = dict()
symbols = [
	'1',
	'2',
	'3Зз',
	'4',
	'5',
	'6Бб',
	'7',
	'8',
	'9',
	'AАaа',
	'bЬьЪъ',
	'GДgд',
	'OОoо0',
	'YУyу',
	'Zz',
	'BВв',
	'CСcс',
	'Dd',
	'EЕЁeеё',
	'Ff',
	'HНhн',
	'Ii',
	'Jj',
	'KКkк',
	'Ll',
	'MМmм',
	'Nn',
	'PРpр',
	'Qq',
	'Rr',
	'Ss',
	'TТtт',
	'UuИЙий',
	'Vv',
	'Ww',
	'XХxх',
	'!',
	'"',
	"'",
	'`',
	'#',
	'$',
	'%',
	'&',
	'([{<',
	')]}>',
	'*',
	'+',
	'.,',
	'-',
	'/',
	':;',
	'=',
	'?',
	'@',
	'\\',
	'^',
	'_',
	'|',
	'~',
	'Гг',
	'Жж',
	'Лл',
	'Пп',
	'Фф',
	'Цц',
	'Чч',
	'ШЩшщ',
	'Ыы',
	'Ээ',
	'Юю',
	'Яя',
]

for index, symbol in enumerate(symbols):
	glyphs = set()
	for char in symbol:
		for font in fonts:
			glyphs.add(font.get_glyph(char))
	symbols_to_glyphs[symbol] = glyphs

for symbol1, glyphs1 in symbols_to_glyphs.items():
	for symbol2, glyphs2 in symbols_to_glyphs.items():
		if symbol1 != symbol2:
			if len(glyphs1 & glyphs2):
				raise Exception("Fucking fuck: " + symbol1 + " and " + symbol2)

char_height = 16
char_width = 8
columns_count = 32
rows_count = 16
image_width = columns_count * char_width
image_height = rows_count * char_height
grid_w = columns_count * 2
grid_h = rows_count * 2
image_shape = (image_height, image_width, 3)
label_shape = (grid_h, grid_w, 1 + 2 + 2 + len(symbols) + len(colors) + len(colors))

def random_shade(color):
	h = random.randrange(color["h"][0], color["h"][1]) % 360 / 360.
	s = random.uniform(color["s"][0], color["s"][1])
	v = random.uniform(color["v"][0], color["v"][1])
	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	return (int(r * 256), int(g * 256), int(b * 256))

def random_colors():
	bg = random.choice(colors)
	while True:
		fg = random.choice(colors)
		if bg != fg:
			break
	return bg, fg

images_count = 10000

def draw_char(image, label, font, left, top, fg, bg, fg_shade, bg_shade):
	symbol = random.choice(symbols)
	char = random.choice(symbol)
	left, top, right, bottom = font.draw(image, char, left=left, top=top, font_color=fg_shade, background_color=bg_shade)
	width = right - left + 1
	height = bottom - top + 1
	x = (left + (width // 2)) / image_width
	y = (top + (height // 2)) / image_height
	grid_x = int(x * grid_w)
	grid_y = int(y * grid_h)
	label[grid_y, grid_x, 0] = 1
	label[grid_y, grid_x, 1] = x * grid_w - grid_x
	label[grid_y, grid_x, 2] = y * grid_h - grid_y
	label[grid_y, grid_x, 3] = width / char_width
	label[grid_y, grid_x, 4] = height / char_height
	label[grid_y, grid_x, 5 + symbols.index(symbol)] = 1
	label[grid_y, grid_x, 5 + len(symbols) + colors.index(fg)] = 1
	label[grid_y, grid_x, 5 + len(symbols) + len(colors) + colors.index(bg)] = 1
	if np.any(label[grid_y, grid_x] > 1):
		print(label[grid_y, grid_x])
		raise Exception("label > 1")
	if np.any(label[grid_y, grid_x] < 0):
		print(label[grid_y, grid_x])
		raise Exception("label < 0")

def generate_example_1():
	bg, fg = random_colors()
	bg_shade, fg_shade = random_shade(bg), random_shade(fg)
	image = np.full(image_shape, bg_shade, np.uint8)
	label = np.zeros(label_shape, np.float32)
	for row in range(1, rows_count - 1, 2):
		font = random.choice(fonts)
		x_offset = random.randint(-3, 3);
		y_offset = random.randint(-7, 7);
		is_word = random.choice([True, False])
		column = 1
		while column < columns_count - 1:
			is_word = not is_word
			if is_word:
				word_length = random.randint(1, min(15, columns_count - 1 - column))
			else:
				word_length = random.randint(1, min(2, columns_count - 1 - column))
			for _ in range(word_length):
				left = column*char_width + x_offset
				top = row*char_height + y_offset
				if is_word:
					draw_char(image, label, font, left, top, fg, bg, fg_shade, None)
				column += 1
	return {
		'image': image,
		'label': label
	}

def generate_example_2():
	bg, fg = random_colors()
	bg_shade, fg_shade = random_shade(bg), random_shade(fg)
	image = np.full(image_shape, (0, 0, 0), np.uint8)
	label = np.zeros(label_shape, np.float32)
	j = 0
	for row in range(rows_count):
		font = random.choice(fonts)
		is_word = random.choice([True, False])
		column = 0
		while column < columns_count:
			is_word = not is_word
			if is_word:
				word_length = random.randint(1, min(15, columns_count - column))
			else:
				word_length = random.randint(1, min(2, columns_count - column))
			for _ in range(word_length):
				if j % 57 == 0:
					bg, fg = random_colors()
					bg_shade, fg_shade = random_shade(bg), random_shade(fg)
				left = column*char_width
				top = row*char_height
				if is_word:
					draw_char(image, label, font, left, top, fg, bg, fg_shade, bg_shade)
				else:
					font.draw(image, ' ', left=left, top=top, font_color=fg_shade, background_color=bg_shade)
				j += 1
				column += 1
	return {
		'image': image,
		'label': label
	}

# example = generate_example_1()
# plt.imshow(example['image'])
# plt.show()
# exit(0)

class Builder(tfds.core.GeneratorBasedBuilder):
	VERSION = tfds.core.Version('0.1.0')

	def _info(self):
		return tfds.core.DatasetInfo(
			builder=self,
			features=tfds.features.FeaturesDict({
				"image": tfds.features.Image(shape=image_shape),
				"label": tfds.features.Tensor(shape=label_shape, dtype=tf.float32),
			})
		)

	def _split_generators(self, dl_manager):
		return [
			tfds.core.SplitGenerator(name=tfds.Split.TRAIN)
		]

	def _generate_examples(self):
		for i in range(images_count):
			if i % 2 == 1:
				yield i, generate_example_1()
			else:
				yield i, generate_example_2()

builder = Builder(data_dir='.')
builder.download_and_prepare()
