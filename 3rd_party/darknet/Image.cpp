
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
	_data = (uint8_t*)stbi__malloc(w*h*c);
}

Image::Image(const std::string& path) {
	_data = stbi_load(path.c_str(), &width, &height, &channels, 0);
	if (!_data) {
		throw std::runtime_error("Cannot load image " + path + " because " + stbi_failure_reason());
	}
}

Image::Image(const uint8_t* data, size_t size) {
	_data = stbi_load_from_memory(data, size, &width, &height, &channels, 0);
	if (!_data) {
		throw std::runtime_error(std::string("Cannot parse image because ") + stbi_failure_reason());
	}
}

Image::~Image() {
	if (_data) {
		stbi_image_free((void*)_data);
	}
}

Image::Image(Image&& other):
	_data(other._data),
	width(other.width),
	height(other.height),
	channels(other.channels)
{
	other._data = nullptr;
	other.width = 0;
	other.height = 0;
	other.channels = 0;
}

Image& Image::operator=(Image&& other) {
	std::swap(_data, other._data);
	std::swap(width, other.width);
	std::swap(height, other.height);
	std::swap(channels, other.channels);
	return *this;
}

void Image::write_png(const std::string& path) {
	if (!stbi_write_png(path.c_str(), width, height, channels, _data, width*channels)) {
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
		_data[i*channels + y1*width*channels + 0] = r;
		_data[i*channels + y2*width*channels + 0] = r;

		_data[i*channels + y1*width*channels + 1] = g;
		_data[i*channels + y2*width*channels + 1] = g;

		_data[i*channels + y1*width*channels + 2] = b;
		_data[i*channels + y2*width*channels + 2] = b;
	}
	for (int i = y1; i <= y2; ++i) {
		_data[x1*channels + i*width*channels + 0] = r;
		_data[x2*channels + i*width*channels + 0] = r;

		_data[x1*channels + i*width*channels + 1] = g;
		_data[x2*channels + i*width*channels + 1] = g;

		_data[x1*channels + i*width*channels + 2] = b;
		_data[x2*channels + i*width*channels + 2] = b;
	}
}

}

namespace darknet {

Image::Image(size_t width, size_t height, size_t channels) {
	w = width;
	h = height;
	c = channels;
	data = (float*)calloc(h*w*c, sizeof(float));
}

Image::Image(const std::string& path): Image(stb::Image(path)) {
}

Image::Image(const stb::Image& stb): Image(stb.width, stb.height, stb.channels) {
	from_stb(stb);
}

Image::~Image() {
	free(data);
	data = nullptr;
}

void Image::from_stb(const stb::Image& stb) {
	if ((w != stb.width) ||
		(h != stb.height) ||
		(c != stb.channels))
	{
		throw std::runtime_error("Invalid image size");
	}
	for (int k = 0; k < c; ++k) {
		for (int j = 0; j < h; ++j) {
			for (int i = 0; i < w; ++i) {
				int dst_index = i + w*j + w*h*k;
				int src_index = k + c*i + c*w*j;
				data[dst_index] = (float)stb._data[src_index]/255.;
			}
		}
	}
}

void Image::to_stb(stb::Image& stb) const {
	if ((w != stb.width) ||
		(h != stb.height) ||
		(c != stb.channels))
	{
		throw std::runtime_error("Invalid image size");
	}
	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < w*h; ++i) {
			stb._data[i*c+k] = (uint8_t)(255*data[i + k*w*h]);
		}
	}
}

void Image::save(const std::string& path) const {
	stb::Image image(w, h, c);
	to_stb(image);
	image.write_png(path);
}

Image::Image(Image&& other):
	w(other.w),
	h(other.h),
	c(other.c),
	data(other.data)
{
	other.data = nullptr;
}

Image& Image::operator=(Image&& other) {
	std::swap(w, other.w);
	std::swap(h, other.h);
	std::swap(c, other.c);
	std::swap(data, other.data);
	return *this;
}

}