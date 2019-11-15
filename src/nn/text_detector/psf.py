
import gzip
import numpy as np

class Glyph:
	def __init__(self, width, height, data):
		self.width = width
		self.height = height
		self.data = data
		self.mask = np.zeros((height, width, 1), np.uint8)
		self.x_min = width
		self.y_min = height
		self.x_max = 0
		self.y_max = 0
		bits_per_line = (width + 7) // 8 * 8
		for y in range(height):
			for x in range(width):
				bit_position = y * bits_per_line + x
				byte_index = bit_position // 8
				bit_index = 7 - bit_position % 8
				if data[byte_index] & (1 << bit_index):
					if self.x_min > x:
						self.x_min = x
					if self.y_min > y:
						self.y_min = y
					if self.x_max < x:
						self.x_max = x
					if self.y_max < y:
						self.y_max = y
					self.mask[y, x] = 1

	def __hash__(self):
		return hash(self.data)

	def __eq__(self, other):
		return self.data == other.data

	def draw(self, image, left, top, font_color, background_color=None):
		fg = self.mask * font_color
		if background_color:
			bg = (1 - self.mask) * background_color
		else:
			bg = (1 - self.mask) * image[top:top+self.height, left:left+self.width]
		image[top:top+self.height, left:left+self.width] = fg + bg
		return left + self.x_min, top + self.y_min, left + self.x_max, top + self.y_max

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

	def set_data(self, i, data):
		assert len(data) == self.charsize
		self.data[i * self.charsize : (i + 1) * self.charsize] = data

	def get_glyph(self, char):
		data = self.get_data(self.unicodes[char])
		return Glyph(self.width, self.height, data)

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
