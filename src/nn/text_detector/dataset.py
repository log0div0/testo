#!/usr/bin/python3

import gzip, os, string, shutil, random, json
import PIL, PIL.ImageDraw
import multiprocessing
import colorsys

class PSF:
	def __init__(self, filename):
		self.load(filename)
		if self.has_unicode:
			self.unicodes = {}
			i = 0
			for chars in self.unicode_data[0:-1].split(self.unicode_term):
				for char in chars.decode('utf-16', 'replace'):
					self.unicodes[char] = i
				i += 1

	def get_data(self, i):
		return self.data[i * self.charsize: (i+1) * self.charsize]

	def get_glyph(self, char):
		return self.get_data(self.unicodes[char])

	def draw(self, image, char, left, top, font_color="white", background_color="black"):
		data = self.get_glyph(char)
		draw = PIL.ImageDraw.Draw(image)
		bits_per_line = (self.width + 7) // 8 * 8
		x_min = self.width
		y_min = self.height
		x_max = 0
		y_max = 0
		for y in range(self.height):
			for x in range(self.width):
				bit_position = y * bits_per_line + x
				byte_index = bit_position // 8
				bit_index = 7 - bit_position % 8
				if data[byte_index] & (1 << bit_index):
					color = font_color
					if x_min > x:
						x_min = x
					if y_min > y:
						y_min = y
					if x_max < x:
						x_max = x
					if y_max < y:
						y_max = y
				else:
					color = background_color
				draw.point((left+x,top+y), color)
		return x_min, y_min, x_max-x_min+1, y_max-y_min+1

	def set_data(self, i, data):
		assert len(data) == self.charsize
		self.data[i * self.charsize : (i + 1) * self.charsize] = data

	def calc(self, offset):
		return self.header2[offset] + (self.header2[offset + 1] << 8) + \
			(self.header2[offset + 2] << 16) + (self.header2[offset + 3] << 24)

	def zap(self, off, v):
		self.header2[off] = v & 255
		self.header2[off + 1] = (v >> 8) & 255
		self.header2[off + 2] = (v >> 16) & 255
		self.header2[off + 3] = (v >> 24) & 255

	def create_header2(self):
		self.zap(0, 0) # version
		self.zap(4, 32) # headersize
		self.zap(8, 0) # flags
		self.zap(12, self.length) # length
		self.zap(16, self.charsize)
		self.zap(20, self.height)
		self.zap(24, self.width)

	def dump_info(self):
		print("type = %d" % self.type)
		print("charsize = %d" % self.charsize)
		print("width = %d" % self.width)
		print("height = %d" % self.height)
		print("length = %d" % self.length)
		print("has_unicode = %s" % self.has_unicode)

	def load_psf_1(self, f):
		self.type = 1
		if self.header[2] & 1 == 1:
			self.length = 512
		else:
			self.length = 256
		if self.header[2] & 2 == 2:
			self.has_unicode = True
		else:
			self.has_unicode = False
		self.height = self.charsize = self.header[3]
		self.width = 8
		self.data = f.read(self.length * self.charsize)
		self.unicode_data = f.read()
		self.unicode_term = b'\xff\xff'

	def load_psf_2(self, f):
		self.type = 2
		self.header2 = f.read(28)
		self.version = self.calc(0)
		self.headersize = self.calc(4)
		self.flags = self.calc(8)
		self.length = self.calc(12)
		self.charsize = self.calc(16)
		self.height = self.calc(20)
		self.width = self.calc(24)
		self.has_unicode = self.flags & 1 == 1
		self.rest_length = self.headersize - 32
		self.rest = f.read(self.rest_length)
		self.data = f.read(self.length * self.charsize)
		self.unicode_data = f.read()
		self.unicode_term = b'\xff'

	def load(self, filename):
		if filename.endswith(".gz"):
			f = gzip.open(filename, "rb")
		else:
			f = open(filename, "rb")
		self.type = 0
		self.header = f.read(4)
		if (self.header[0] == 0x36 and self.header[1] == 0x04):
			self.load_psf_1(f)
		elif (self.header[0] == 0x72 and self.header[1] == 0xb5 and self.header[2] == 0x4a and self.header[3] == 0x86):
			self.load_psf_2(f)
		f.close()

	def save(self, filename):
		if filename.endswith(".gz"):
			f = gzip.open(filename, "wb")
		else:
			f = open(filename, "wb")
		if self.type == 1:
			f.write(self.header)
			f.write(self.data)
			f.write(self.unicode_data)
		elif self.type == 2:
			f.write(self.header)
			f.write(self.header2)
			f.write(self.rest)
			f.write(self.data)
			f.write(self.unicode_data)
		f.close()

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

chars = [char for char in string.printable if not char.isspace()]
for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
	chars.append(char)
	chars.append(char.upper())

fonts = [PSF(os.path.join(fonts_dir, font_name)) for font_name in font_names]
symbols_to_glyphs = dict()
chars_to_symbols = dict()
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
		chars_to_symbols[char] = index
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
	char = random.choice([random.choice(chars), ' '])
	x, y, width, height = font.draw(image, char, left=left, top=top, font_color=foreground["rgb"], background_color=background["rgb"])
	x_center = (left + x + (width // 2)) / image_width
	y_center = (top + y + (height // 2)) / image_height
	if char != ' ':
		return "%s %s %s %s %s %s %s\n" % (x_center, y_center, (width + 2) / image_width, (height + 2) / image_height,
			chars_to_symbols[char], foreground["index"], background["index"])
	else:
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
		image = PIL.Image.new("RGB", (image_width, image_height), background["rgb"])
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
		image = PIL.Image.new("RGB", (image_width, image_height), background["rgb"])
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

	image.save(image_path)
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
