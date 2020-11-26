
#pragma once

#include <string>
#include <algorithm>
#include <stdexcept>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <string.h>

namespace stb {

#ifdef WIN32
#define __PACKED__
#pragma pack(push,1)
#else
#define __PACKED__ __attribute__((__packed__))
#endif

struct RGBA {
	uint8_t r, g, b, a;
} __PACKED__;

struct RGB {
	uint8_t r, g, b;

	static RGB white() {
		return {255, 255, 255};
	}

	static RGB black() {
		return {0, 0, 0};
	}

	bool operator==(const RGB& other) const {
		return
			(r == other.r) &&
			(g == other.g) &&
			(b == other.b);
	}

	bool operator!=(const RGB& other) const {
		return
			(r != other.r) ||
			(g != other.g) ||
			(b != other.b);
	}

	const uint8_t& operator[](uint8_t index) const {
		uint8_t* data = (uint8_t*)this;
		return data[index];
	}

	uint8_t& operator[](uint8_t index) {
		uint8_t* data = (uint8_t*)this;
		return data[index];
	}

	int max_channel_diff(const RGB& other) const {
		return
			std::max(
				std::max(
					std::abs(int(r) - int(other.r)),
					std::abs(int(g) - int(other.g))
				),
				std::abs(int(b) - int(other.b))
			);
	}
} __PACKED__;

#ifdef WIN32
#pragma pack(pop)
#endif

template <typename Pixel>
struct Image {
	static constexpr int c = sizeof(Pixel);

	Image() {}

	Image(int w_, int h_) {
		w = w_;
		h = h_;
		data = (uint8_t*)malloc(w*h*c);
		if (!data) {
			throw std::runtime_error("malloc failed");
		}
	}

	Image(int w_, int h_, Pixel pixel): Image(w_, h_) {
			fill(pixel);
	}

	Image(const char* path) {
		int actual_c = 0;
		data = stbi_load(path, &w, &h, &actual_c, c);
		if (!data) {
			throw std::runtime_error("Cannot load image " + std::string(path) + " because " + stbi_failure_reason());
		}
	}

	Image(const std::string& path): Image(path.c_str()) {}

	Image(const uint8_t* buffer, size_t buffer_len) {
		int actual_c = 0;
		data = stbi_load_from_memory(buffer, buffer_len, &w, &h, &actual_c, c);
		if (!data) {
			throw std::runtime_error(std::string("Cannot parse image because ") + stbi_failure_reason());
		}
	}

	~Image() {
		if (data) {
			stbi_image_free(data);
			data = nullptr;
		}
	}

	Image(const Image& other): Image(other.w, other.h) {
		memcpy(data, other.data, other.data_len());
	}

	Image& operator=(const Image& other) {
		uint8_t* new_data = (uint8_t*)realloc(data, other.data_len());
		if (!new_data) {
			throw std::runtime_error("realloc failed");
		}
		data = new_data;
		w = other.w;
		h = other.h;
		memcpy(data, other.data, other.data_len());
		return *this;
	}

	Image(Image&& other):
		data(other.data),
		w(other.w),
		h(other.h)
	{
		other.data = nullptr;
		other.w = 0;
		other.h = 0;
	}

	Image& operator=(Image&& other) {
		std::swap(data, other.data);
		std::swap(w, other.w);
		std::swap(h, other.h);
		return *this;
	}

	void fill(Pixel p) {
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				at(x, y) = p;
			}
		}
	}

	Pixel& at(int x, int y) {
		return *(Pixel*)&data[y*w*c + x*c];
	}

	const Pixel& at(int x, int y) const {
		return *(Pixel*)&data[y*w*c + x*c];
	}

	void write_png(const std::string& path) {
		if (!stbi_write_png(path.c_str(), w, h, c, data, w*c)) {
			throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
		}
	}

	void write_jpg(const std::string& path, int quality) {
		if (!stbi_write_jpg(path.c_str(), w, h, c, data, quality)) {
			throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
		}
	}

	size_t pixels_count() const {
		return w * h;
	}

	size_t data_len() const {
		return w * h * c;
	}

	size_t stride() const {
		return w * c;
	}

	Image resize(int out_w, int out_h) const {
		Image result(out_w, out_h);
		if (!stbir_resize_uint8(
			this->data, this->w, this->h, this->stride(),
			result.data, result.w, result.h, result.stride(),
			sizeof(Pixel))
		) {
			throw std::runtime_error("stbir_resize_uint8 failed");
		}
		return result;
	}

	Image sub_img(int off_x, int off_y, int sub_w, int sub_h) const {
		if (off_x < 0) {
			throw std::runtime_error("off_x < 0");
		}
		if (off_y < 0) {
			throw std::runtime_error("off_y < 0");
		}
		if ((off_x + sub_w) > w) {
			throw std::runtime_error("(off_x + sub_w) > w");
		}
		if ((off_y + sub_h) > h) {
			throw std::runtime_error("(off_y + sub_h) > h");
		}
		Image sub(sub_w, sub_h);
		for (int y = 0; y < sub_h; ++y) {
			for (int x = 0; x < sub_w; ++x) {
				sub.at(x, y) = at(off_x + x, off_y + y);
			}
		}
		return sub;
	}

	uint8_t* data = nullptr;
	int w = 0;
	int h = 0;
};

}
