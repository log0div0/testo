
import gzip
import argparse
import os
from PIL import Image, ImageDraw

common = '''0123456789!?"'#$%&@()[]{}<>+-*/\\.,:;^~=|_'''
english = 'abcdefghijklmnopqrstuvwxyz'
ENGLISH = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
russian = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
RUSSIAN = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
alphabet = common + english + ENGLISH + russian + RUSSIAN

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

	def draw(self, char):
		image = Image.new("RGBA", (self.width, self.height))
		draw = ImageDraw.Draw(image)
		data = self.get_data(self.unicodes[char])
		bits_per_line = (self.width + 7) // 8 * 8
		for y in range(self.height):
			for x in range(self.width):
				bit_position = y * bits_per_line + x
				byte_index = bit_position // 8
				bit_index = 7 - bit_position % 8
				if data[byte_index] & (1 << bit_index):
					draw.point([x, y], fill=(255,255,255,255))
		return image

	def calc(self, offset):
		return self.header2[offset] + (self.header2[offset + 1] << 8) + \
			(self.header2[offset + 2] << 16) + (self.header2[offset + 3] << 24)

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

parser = argparse.ArgumentParser()
parser.add_argument('in_path')
parser.add_argument('out_path')
args = parser.parse_args()

psf = PSF(args.in_path)

psf.dump_info()

if not os.path.exists(args.out_path):
	os.mkdir(args.out_path)

for i, char in enumerate(alphabet):
	image = psf.draw(char)
	image.save(os.path.join(args.out_path, str(i) + ".png"))
