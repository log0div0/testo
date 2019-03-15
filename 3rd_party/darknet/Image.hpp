
#pragma once

#include "include/darknet.h"
#include <string>

namespace darknet {

struct Image {
	Image() = default;
	Image(size_t width, size_t height, size_t channels);
	Image(const std::string& path);
	~Image();

	Image(const Image&) = delete;
	Image& operator=(const Image&) = delete;
	Image(Image&& other);
	Image& operator=(Image&& other);

	void save(const std::string& path) const;

	size_t width() const {
		return impl.w;
	}

	size_t height() const {
		return impl.h;
	}

	size_t channels() const {
		return impl.c;
	}

	float& operator[](size_t index) {
		return impl.data[index];
	}

	const float& operator[](size_t index) const {
		return impl.data[index];
	}

	Image letterbox(size_t width, size_t height) const;

	float* data() {
		return impl.data;
	}

	const float* data() const {
		return impl.data;
	}

	void draw(box box, float r, float g, float b);

	image impl = {};

private:
	Image(const image& impl_): impl(impl_) {}
};

}
