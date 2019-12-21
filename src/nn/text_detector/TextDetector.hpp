
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include <memory>

namespace Ort {
class Env;
class Session;
class Value;
}

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
	TextDetector();
	~TextDetector();

	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	std::vector<Rect> detect(stb::Image& image);

private:
	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;

	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};
