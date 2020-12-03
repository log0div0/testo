
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include <map>

namespace psf {

struct Font {
	Font(const std::string& path);

	int height() const {
		return _height;
	}
	int width() const {
		return _width;
	}

	stb::Image<uint8_t> codepointBitmap(char32_t codepoint) const;

private:
	uint32_t _height = 0;
	uint32_t _width = 0;
	uint32_t _charsize = 0;
	std::vector<uint8_t> _data;
	std::map<char32_t, size_t> _codepoint_map;
};

}
