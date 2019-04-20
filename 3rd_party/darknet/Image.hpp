
#pragma once

#include "include/darknet.h"
#include <string>

namespace stb {

struct Image {
	Image() {}
	Image(int w, int h, int c);
	Image(const std::string& path);
	Image(const uint8_t* data, size_t size);
	~Image();

	Image(const Image& other) = delete;
	Image& operator=(const Image& other) = delete;
	Image(Image&& other);
	Image& operator=(Image&& other);

	void write_png(const std::string& path);

	const uint8_t* data() const {
		return _data;
	}

	uint8_t* data() {
		return _data;
	}

	size_t size() const {
		return width * height * channels;
	}

	uint8_t* _data = nullptr;
	int width = 0;
	int height = 0;
	int channels = 0;
};

}

namespace darknet {

struct Image {
	Image() = default;
	Image(size_t width, size_t height, size_t channels);
	Image(const stb::Image& stb);
	Image(const std::string& path);
	~Image();

	Image(const Image&) = delete;
	Image& operator=(const Image&) = delete;
	Image(Image&& other);
	Image& operator=(Image&& other);

	void from_stb(const stb::Image& stb);
	void to_stb(stb::Image& stb) const;

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

	void draw(int left, int top, int right, int bottom, float r, float g, float b);

	int w = 0;
	int h = 0;
	int c = 0;
	float* data = nullptr;
};

}
