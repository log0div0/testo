
#include "Font.hpp"
#include <zlib.h>
#include <iostream>
#include <codecvt>
#include <locale>

namespace psf {

struct GZFile {
	GZFile(const std::string& path) {
		f = gzopen(path.c_str(), "rb");
		if (f == Z_NULL) {
			throw std::runtime_error("gzopen failed");
		}
	}
	~GZFile() {
		if (f != Z_NULL) {
			gzclose(f);
			f = Z_NULL;
		}
	}

	uint32_t read(uint8_t* buf, uint32_t len) {
		int res = gzread(f, buf, len);
		if (res <= 0) {
			if (gzeof(f)) {
				return 0;
			} else {
				throw std::runtime_error("gz error");
			}
		}
		return res;
	}

	uint8_t read_u8() {
		uint8_t val = 0;
		read((uint8_t*)&val, sizeof(val));
		return val;
	}

	uint16_t read_u16() {
		uint16_t val = 0;
		read((uint8_t*)&val, sizeof(val));
		return val;
	}

	uint32_t read_u32() {
		uint32_t val = 0;
		read((uint8_t*)&val, sizeof(val));
		return val;
	}

	std::vector<uint8_t> read_rest() {
		std::vector<uint8_t> result;
		uint8_t bytes[512];
		while (true) {
			uint32_t bytes_read = read(bytes, 512);
			if (!bytes_read) {
				break;
			}
			result.insert(result.end(), bytes, bytes + bytes_read);
		}
		return result;
	}

	gzFile f = Z_NULL;
};

#define PSF1_MODE512    0x01
#define PSF1_MODEHASTAB 0x02

std::string to_utf8(const std::u16string &s){
	std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> conv;
	return conv.to_bytes(s);
}

std::u32string to_utf32(const std::string &s) {
	std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
	return conv.from_bytes(s);
}

Font::Font(const std::string& path) {
	GZFile file(path);

	uint16_t magic = file.read_u16();
	if (magic == 0x0436) {
		// v1
		uint8_t mode = file.read_u8();
		uint32_t length = 0;
		if (mode & PSF1_MODE512) {
			length = 512;
		} else {
			length = 256;
		}
		if (!(mode & PSF1_MODEHASTAB)) {
			throw std::runtime_error("NO UNICODE TABLE");
		}
		_height = file.read_u8();
		_width = 8;
		_charsize = _height;
		_data.resize(length * _charsize);
		file.read(_data.data(), _data.size());
		std::vector<uint8_t> unicode_data = file.read_rest();
		size_t begin = 0, end = 0;
		uint32_t gliph_index = 0;
		while (begin != unicode_data.size()) {
			while ((unicode_data.at(end) != 0xff) || (unicode_data.at(end + 1) != 0xff)) {
				++end;
			}
			size_t codepoint_count = (end - begin) / 2;
			std::u16string u16str((char16_t*)(unicode_data.data() + begin), codepoint_count);
			try {
				std::string str = to_utf8(u16str);
				std::u32string u32str = to_utf32(str);
				for (char32_t codepoint: u32str) {
					_codepoint_map[codepoint] = gliph_index;
				}
			} catch (const std::exception&) {
				// skip
			}
			end += 2;
			begin = end;
			++gliph_index;
		}
	} else if (magic == 0xb572) {
		// v2
		magic = file.read_u16();
		if (magic != 0x864a) {
			throw std::runtime_error("Invalid v2 magic");
		}
		uint32_t version = file.read_u32();
		(void)version;
		uint32_t header_size = file.read_u32();
		uint32_t flags = file.read_u32();
		uint32_t length = file.read_u32();
		_charsize = file.read_u32();
		_height = file.read_u32();
		_width = file.read_u32();
		if (!(flags & 1)) {
			throw std::runtime_error("NO UNICODE TABLE");
		}
		uint32_t rest_hdr_len = header_size - 32;
		if (rest_hdr_len) {
			std::vector<uint8_t> tmp;
			tmp.resize(rest_hdr_len);
			file.read(tmp.data(), tmp.size());
		}
		_data.resize(length * _charsize);
		file.read(_data.data(), _data.size());
		std::vector<uint8_t> unicode_data = file.read_rest();
		size_t begin = 0, end = 0;
		uint32_t gliph_index = 0;
		while (begin != unicode_data.size()) {
			while (unicode_data.at(end) != 0xff) {
				++end;
			}
			size_t codepoint_count = (end - begin) / 2;
			std::u16string u16str((char16_t*)(unicode_data.data() + begin), codepoint_count);
			std::string str = to_utf8(u16str);
			std::cout << gliph_index << std::endl;
			std::cout << str << std::endl;
			std::u32string u32str = to_utf32(str);
			for (char32_t codepoint: u32str) {
				_codepoint_map[codepoint] = gliph_index;
			}
			end += 1;
			begin = end;
			++gliph_index;
		}
	} else {
		throw std::runtime_error("Not a psf file");
	}
}

stb::Image<uint8_t> Font::codepointBitmap(char32_t codepoint) const {
	size_t gliph_index = _codepoint_map.at(codepoint);
	stb::Image<uint8_t> bitmap(_width, _height);
	bitmap.fill(0);
	size_t gliph_off = gliph_index * _charsize;
	size_t bits_per_line = (_width + 7) / 8 * 8;
	for (uint32_t y = 0; y < _height; ++y) {
		for (uint32_t x = 0; x < _width; ++x) {
			size_t bit_position = y * bits_per_line + x;
			size_t byte_index = gliph_off + bit_position / 8;
			size_t bit_index = 7 - bit_position % 8;
			if (_data.at(byte_index) & (1 << bit_index)) {
				bitmap.at(x, y) = 255;
			}
		}
	}
	return bitmap;
}

}
