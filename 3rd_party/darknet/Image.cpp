
#include "Image.hpp"
#include <algorithm>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace darknet {

Image::Image(size_t width, size_t height, size_t channels) {
	w = width;
	h = height;
	c = channels;
	data = (float*)calloc(h*w*c, sizeof(float));
}

Image::Image(const std::string& path) {
	uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 0);
	if (!data) {
		throw std::runtime_error("Cannot load image " + path + " because " + stbi_failure_reason());
	}
	this->data = (float*)calloc(h*w*c, sizeof(float));
	for (int k = 0; k < c; ++k) {
		for (int j = 0; j < h; ++j) {
			for (int i = 0; i < w; ++i) {
				int dst_index = i + w*j + w*h*k;
				int src_index = k + c*i + c*w*j;
				this->data[dst_index] = (float)data[src_index]/255.;
			}
		}
	}
	free(data);
}

Image::~Image() {
	free(data);
	data = nullptr;
}

void Image::save(const std::string& path) const {
	uint8_t* data = (uint8_t*)calloc(w*h*c, sizeof(uint8_t));
	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < w*h; ++i) {
			data[i*c+k] = (uint8_t)(255*this->data[i + k*w*h]);
		}
	}
	int success = stbi_write_png(path.c_str(), w, h, c, data, w*c);
	free(data);
	if (!success) {
		throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
	}
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

void Image::draw(int x1, int y1, int x2, int y2, float r, float g, float b) {
	if (x1 < 0) x1 = 0;
	if (x1 >= w) x1 = w-1;
	if (x2 < 0) x2 = 0;
	if (x2 >= w) x2 = w-1;

	if (y1 < 0) y1 = 0;
	if (y1 >= h) y1 = h-1;
	if (y2 < 0) y2 = 0;
	if (y2 >= h) y2 = h-1;

	for (int i = x1; i <= x2; ++i) {
		data[i + y1*w + 0*w*h] = r;
		data[i + y2*w + 0*w*h] = r;

		data[i + y1*w + 1*w*h] = g;
		data[i + y2*w + 1*w*h] = g;

		data[i + y1*w + 2*w*h] = b;
		data[i + y2*w + 2*w*h] = b;
	}
	for (int i = y1; i <= y2; ++i) {
		data[x1 + i*w + 0*w*h] = r;
		data[x2 + i*w + 0*w*h] = r;

		data[x1 + i*w + 1*w*h] = g;
		data[x2 + i*w + 1*w*h] = g;

		data[x1 + i*w + 2*w*h] = b;
		data[x2 + i*w + 2*w*h] = b;
	}
}

}