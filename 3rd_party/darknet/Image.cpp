
#include "Image.hpp"
#include <algorithm>
#include <image.h>

namespace darknet {

Image::Image(size_t width, size_t height, size_t channels): image(make_image(width, height, channels)) {
}

Image::Image(const std::string& path): image(load_image((char*)path.c_str(), 0, 0, 3)) {
}

Image::~Image() {
	free_image(*this);
}

void Image::save(const std::string& path) const {
	save_image(*this, (char*)path.c_str());
}

Image::Image(Image&& other): image(other) {
	other.data = nullptr;
}

Image& Image::operator=(Image&& other) {
	std::swap(*(image*)this, *(image*)&other);
	return *this;
}

Image Image::letterbox(size_t width, size_t height) const {
	return letterbox_image(*this, width, height);
}

void Image::draw(uint16_t left, uint16_t top, uint16_t right, uint16_t bottom, float r, float g, float b) {
	draw_box(*this, left, top, right, bottom, r, g, b);
}

}