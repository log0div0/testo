
#pragma once

#include "include/darknet.h"
#include <string>

namespace darknet {

struct Image: image {
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
		return w;
	}

	size_t height() const {
		return h;
	}

	size_t channels() const {
		return c;
	}

	float& operator[](size_t index) {
		return data[index];
	}

	const float& operator[](size_t index) const {
		return data[index];
	}

	Image letterbox(size_t width, size_t height) const;

	void draw(uint16_t left, uint16_t top, uint16_t right, uint16_t bottom, float r, float g, float b);

private:
	Image(const image& impl_): image(impl_) {}
};

}
