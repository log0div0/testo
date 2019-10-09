
#include "Image.hpp"
#include <algorithm>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace stb {

Image::Image(int w, int h, int c) {
	width = w;
	height = h;
	channels = c;
	data = (uint8_t*)stbi__malloc(w*h*c);
}

Image::Image(const std::string& path) {
	data = stbi_load(path.c_str(), &width, &height, &channels, 0);
	if (!data) {
		throw std::runtime_error("Cannot load image " + path + " because " + stbi_failure_reason());
	}
}

Image::Image(const uint8_t* buffer, size_t buffer_len) {
	data = stbi_load_from_memory(buffer, buffer_len, &width, &height, &channels, 0);
	if (!data) {
		throw std::runtime_error(std::string("Cannot parse image because ") + stbi_failure_reason());
	}
}

Image::~Image() {
	if (data) {
		stbi_image_free((void*)data);
	}
}

Image::Image(Image&& other):
	data(other.data),
	width(other.width),
	height(other.height),
	channels(other.channels)
{
	other.data = nullptr;
	other.width = 0;
	other.height = 0;
	other.channels = 0;
}

Image& Image::operator=(Image&& other) {
	std::swap(data, other.data);
	std::swap(width, other.width);
	std::swap(height, other.height);
	std::swap(channels, other.channels);
	return *this;
}

void Image::write_png(const std::string& path) {
	if (!stbi_write_png(path.c_str(), width, height, channels, data, width*channels)) {
		throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
	}
}

void Image::draw(int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b) {
	if (x1 < 0) x1 = 0;
	if (x1 >= width) x1 = width - 1;
	if (x2 < 0) x2 = 0;
	if (x2 >= width) x2 = width - 1;

	if (y1 < 0) y1 = 0;
	if (y1 >= height) y1 = height - 1;
	if (y2 < 0) y2 = 0;
	if (y2 >= height) y2 = height - 1;

	for (int i = x1; i <= x2; ++i) {
		data[i*channels + y1*width*channels + 0] = r;
		data[i*channels + y2*width*channels + 0] = r;

		data[i*channels + y1*width*channels + 1] = g;
		data[i*channels + y2*width*channels + 1] = g;

		data[i*channels + y1*width*channels + 2] = b;
		data[i*channels + y2*width*channels + 2] = b;
	}
	for (int i = y1; i <= y2; ++i) {
		data[x1*channels + i*width*channels + 0] = r;
		data[x2*channels + i*width*channels + 0] = r;

		data[x1*channels + i*width*channels + 1] = g;
		data[x2*channels + i*width*channels + 1] = g;

		data[x1*channels + i*width*channels + 2] = b;
		data[x2*channels + i*width*channels + 2] = b;
	}
}

}
