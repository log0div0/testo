
#include "Image.hpp"
#include <algorithm>

namespace darknet {

Image::Image(size_t width, size_t height, size_t channels) {
	impl = make_image(width, height, channels);
}

Image::~Image() {
	free_image(impl);
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

}