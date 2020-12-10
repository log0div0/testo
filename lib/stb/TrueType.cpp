
#include "TrueType.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

#include <fstream>

namespace stb {

std::vector<uint8_t> readFile(const std::string& path)
{
	// open the file:
	std::streampos fileSize;
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}

	// get its size:
	file.seekg(0, std::ios::end);
	fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	// read the data:
	std::vector<uint8_t> fileData(fileSize);
	file.read((char*) &fileData[0], fileSize);
	return fileData;
}

Font::Font(const std::string& path) {
	buf = readFile(path);
	if (!stbtt_InitFont(&info, buf.data(), 0)) {
		throw std::runtime_error("stbtt_InitFont failed");
	}
}

Font::~Font() {

}

}
