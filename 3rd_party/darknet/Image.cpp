
#include "Image.hpp"
#include <algorithm>
#include <image.h>

namespace darknet {

Image::Image(size_t width, size_t height, size_t channels) {
	impl = make_image(width, height, channels);
}

Image::Image(const std::string& path) {
	impl = load_image((char*)path.c_str(), 0, 0, 3);
}

Image::~Image() {
	free_image(impl);
}

void Image::save(const std::string& path) const {
	save_image(impl, path.c_str());
}

Image::Image(Image&& other): impl(other.impl) {
	other.impl = {};
}

Image& Image::operator=(Image&& other) {
	std::swap(impl, other.impl);
	return *this;
}

Image Image::letterbox(size_t width, size_t height) const {
	return letterbox_image(impl, width, height);
}

void Image::draw(box box, float r, float g, float b) {
	draw_bbox(impl, box, 1, r, g, b);
}

}