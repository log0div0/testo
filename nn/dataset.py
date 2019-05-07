
import gzip, os, string, shutil, random, json
import PIL, PIL.ImageDraw

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

font_charset = 'Uni2'
font_names = ['Fixed', 'VGA']
font_size = '16'

colors = [
	"White",
	"Silver",
	"Gray",
	"Black",
	"Red",
	"Maroon",
	"Yellow",
	"Olive",
	"Lime",
	"Green",
	"Aqua",
	"Teal",
	"Blue",
	"Navy",
	"Fuchsia",
	"Purple"
]

fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

chars = [char for char in string.printable if not char.isspace()]
for char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя":
	chars.append(char)
	chars.append(char.upper())

fonts = [PSF(os.path.join(fonts_dir, font_charset + '-' + font_name + font_size + '.psf.gz')) for font_name in font_names]
chars_to_glyphs = dict()
chars_to_symbols = dict()
symbols = list()
for char in chars:
	glyphs = set()
	for font in fonts:
		glyphs.add(font.get_glyph(char))
	for char2, glyphs2 in chars_to_glyphs.items():
		intersection = glyphs & glyphs2
		if len(intersection):
			if glyphs != glyphs2:
				raise Exception("Fucking fuck: " + char + " and " + char2)
			else:
				symbol_code = chars_to_symbols[char2]
				chars_to_symbols[char] = symbol_code
				symbols[symbol_code] += char
				break
	chars_to_glyphs[char] = glyphs
	if char not in chars_to_symbols:
		chars_to_symbols[char] = len(symbols)
		symbols.append(char)

print("Symbols count: ", len(symbols))

char_height = 16
char_width = 8
columns_count = 32
rows_count = 16
image_width = columns_count * char_width
image_height = rows_count * char_height

def random_colors():
	background = random.choice(colors)
	while True:
		foreground = random.choice(colors)
		if foreground != background:
			break
	return background, foreground

def draw_char(image, left, top, foreground, background, font):
	char = random.choice([random.choice(chars), ' '])
	x, y, width, height = font.draw(image, char, left=left, top=top, font_color=foreground, background_color=background)
	x_center = (left + x + (width // 2)) / image_width
	y_center = (top + y + (height // 2)) / image_height
	if char != ' ':
		return "%s %s %s %s %s\n" % (chars_to_symbols[char], x_center, y_center, (width + 2) / image_width, (height + 2) / image_height)
	else:
		return ""

def main(base_dir, image_count):
	images_dir = os.path.join(base_dir, "images")
	os.mkdir(images_dir)
	labels_dir = os.path.join(base_dir, "labels")
	os.mkdir(labels_dir)

	for image_index in range(image_count):
		print(str(image_index) + '/' + str(image_count))

		image_path = os.path.join(images_dir, str(image_index) + '.png')
		label_path = os.path.join(labels_dir, str(image_index) + '.txt')

		if image_index % 4 < 3:
			background, foreground = random_colors()
			image = PIL.Image.new("RGB", (image_width, image_height), background)
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
			image = PIL.Image.new("RGB", (image_width, image_height), background)
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

	dataset_file_path = os.path.join(base_dir, "dataset.json")
	with open(dataset_file_path, 'w') as dataset_file:
		dataset_file.write(json.dumps({
			"image_count": image_count,
			"image_width": image_width,
			"image_height": image_height,
			"images_dir": images_dir,
			"labels_dir": labels_dir
		}))

	symbols_file_path = os.path.join(base_dir, "symbols.json")
	with open(symbols_file_path, 'w') as symbols_file:
		symbols_file.write(json.dumps(symbols))

if __name__ == "__main__":
	dataset_dir = os.path.join(os.getcwd(), "dataset")
	if os.path.exists(dataset_dir):
		shutil.rmtree(dataset_dir)
	os.mkdir(dataset_dir)

	main(dataset_dir, 10000)
