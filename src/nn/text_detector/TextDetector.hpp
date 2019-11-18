
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include <memory>

namespace Ort {
class Env;
class Session;
class Value;
}

struct Box {
	float x, y, h, w;
	float uio(const Box& other) const {
		return intersection(other)/union_(other);
	}
	float intersection(const Box& other) const {
		float w = overlap(this->x, this->w, other.x, other.w);
		float h = overlap(this->y, this->h, other.y, other.h);
		if(w < 0 || h < 0) {
			return 0;
		}
		float area = w*h;
		return area;
	}
	float union_(const Box& other) const {
		float i = intersection(other);
		float u = w*h + other.w*other.h - i;
		return u;
	}

	static float overlap(float x1, float w1, float x2, float w2)
	{
		float l1 = x1 - w1/2;
		float l2 = x2 - w2/2;
		float left = l1 > l2 ? l1 : l2;
		float r1 = x1 + w1/2;
		float r2 = x2 + w2/2;
		float right = r1 < r2 ? r1 : r2;
		return right - left;
	}
};

struct Rect {
	int32_t left = 0, top = 0, right = 0, bottom = 0;

	float iou(const Rect& other) const {
		return float((*this & other).area()) / (*this | other).area();
	}

	int32_t area() const {
		return (right - left) * (bottom - top);
	}

	Rect operator|(const Rect& other) const {
		return {
			std::min(left, other.left),
			std::min(top, other.top),
			std::max(right, other.right),
			std::max(bottom, other.bottom)
		};
	}
	Rect& operator|=(const Rect& other) {
		left = std::min(left, other.left);
		top = std::min(top, other.top);
		right = std::max(right, other.right);
		bottom = std::max(bottom, other.bottom);
		return *this;
	}
	Rect operator&(const Rect& other) const {
		if (left > other.right) {
			return {};
		}
		if (top > other.bottom) {
			return {};
		}
		if (right < other.left) {
			return {};
		}
		if (bottom < other.top) {
			return {};
		}
		return {
			std::max(left, other.left),
			std::max(top, other.top),
			std::min(right, other.right),
			std::min(bottom, other.bottom)
		};
	}
	Rect& operator&=(const Rect& other);

	int32_t width() const {
		return right - left;
	}

	int32_t height() const {
		return bottom - top;
	}
};

struct TextDetector {
	~TextDetector();

	std::vector<Rect> detect(stb::Image& image,
		const std::string& text,
		const std::string& foreground = {},
		const std::string& background = {});

	static TextDetector& instance()
	{
	        static TextDetector text_detector;
	        return text_detector;
	}

private:
	TextDetector();
	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;
	bool find_substr(int left, int top,
		const std::vector<std::string>& query, size_t index,
		int foreground_id, int background_id,
		int foreground_hits, int background_hits,
		std::vector<Rect>& rects
	);

	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;

	float at(int x, int y, int c);
};
