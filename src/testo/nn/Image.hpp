
#pragma once

#include <stb_image.h>
#include <string>

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

	size_t size() const {
		return width * height * channels;
	}

	void draw(int left, int top, int right, int bottom, uint8_t r, uint8_t g, uint8_t b);

	uint8_t* data = nullptr;
	int width = 0;
	int height = 0;
	int channels = 0;
};
