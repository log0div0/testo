#!/usr/bin/python3

import os, string, shutil, random, json
import multiprocessing
import colorsys
import PIL.Image
import numpy as np
from psf import PSF

font_names = [
	'Uni2-Continent.psf.gz',
	'Uni2-Continent-FirstScreen.psf.gz',
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
	'Gg',
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
	'Uu',
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
	'Дд',
	'Жж',
	'ИЙий',
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
columns_count = 64
rows_count = 16
image_width = columns_count * char_width
image_height = rows_count * char_height

def color_diff(a, b):
	return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def random_color():
	index = random.randrange(len(colors))
	color = colors[index]
	h = random.randrange(color["h"][0], color["h"][1]) % 360 / 360.
	s = random.uniform(color["s"][0], color["s"][1])
	v = random.uniform(color["v"][0], color["v"][1])
	r, g, b = colorsys.hsv_to_rgb(h, s, v)
	return {
		"index": index,
		"rgb": (int(r * 256), int(g * 256), int(b * 256))
	}

def random_colors():
	background = random_color()
	while True:
		foreground = random_color()
		if color_diff(foreground["rgb"], background["rgb"]) > 350:
			break
	return background, foreground

def draw_char(image, left, top, foreground, background, font):
	symbol = random.choice([random.choice(symbols), None])
	if symbol:
		char = random.choice(symbol)
		x, y, width, height = font.draw(image, char, left=left, top=top, font_color=foreground["rgb"], background_color=background["rgb"])
		x_center = (left + x + (width // 2)) / image_width
		y_center = (top + y + (height // 2)) / image_height
		return "%s %s %s %s %s %s %s\n" % (x_center, y_center, (width + 2) / image_width, (height + 2) / image_height,
			symbols.index(symbol), foreground["index"], background["index"])
	else:
		font.draw(image, ' ', left=left, top=top, font_color=foreground["rgb"], background_color=background["rgb"])
		return ""

dataset_dir = os.path.join(os.getcwd(), "dataset")
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

images_count = 20000

def main(image_index):
	image_path = os.path.join(images_dir, str(image_index) + '.png')
	label_path = os.path.join(labels_dir, str(image_index) + '.txt')

	background, foreground = random_colors()
	if image_index % 4 < 3:
		image = np.full((image_height, image_width, 3), background["rgb"], np.uint8)
		label = ""
		for row in range(1, rows_count - 1, 3):
			font = random.choice(fonts)
			x_offset = random.randint(-3, 3);
			y_offset = random.randint(-7, 7);
			for column in range(1, columns_count - 1):
				left = column*char_width + x_offset
				top = row*char_height + y_offset
				label += draw_char(image, left, top, foreground, background, font)
	else:
		image = np.full((image_height, image_width, 3), background["rgb"], np.uint8)
		label = ""
		j = 0
		for row in range(rows_count):
			font = random.choice(fonts)
			for column in range(columns_count):
				if j % 57 == 0:
					background, foreground = random_colors()
				left = column*char_width
				top = row*char_height
				label += draw_char(image, left, top, foreground, background, font)
				j += 1

	PIL.Image.fromarray(image).save(image_path)
	with open(label_path, "w") as file:
		file.write(label)

if __name__ == "__main__":
	if os.path.exists(dataset_dir):
		shutil.rmtree(dataset_dir)
	os.mkdir(dataset_dir)

	os.mkdir(images_dir)
	os.mkdir(labels_dir)

	with multiprocessing.Pool() as p:
		for x, _ in enumerate(p.imap(main, range(images_count))):
			print(x, "/", images_count)

	symbols_file_path = os.path.join(dataset_dir, "symbols.json")
	with open(symbols_file_path, 'w') as symbols_file:
		symbols_file.write(json.dumps(symbols))
