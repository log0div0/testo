
#pragma once

#include "nlohmann/json.hpp"

#include "Point.hpp"
#include <algorithm>

namespace nn {

struct Rect {
	int32_t left = 0, top = 0, right = -1, bottom = -1;

	float iou(const Rect& other) const {
		return float((*this & other).area()) / (*this | other).area();
	}

	int32_t area() const {
		return width() * height();
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
		return right - left + 1;
	}

	int32_t height() const {
		return bottom - top + 1;
	}

	int32_t center_x() const {
		return left + width() / 2;
	}

	int32_t center_y() const {
		return top + height() / 2;
	}

	Point left_top() const {
		return {left, top};
	}

	Point left_bottom() const {
		return {left, bottom};
	}

	Point right_top() const {
		return {right, top};
	}

	Point right_bottom() const {
		return {right, bottom};
	}

	Point center() const {
		return {center_x(), center_y()};
	};

	Point center_top() const {
		return {center_x(), top};
	}

	Point center_bottom() const {
		return {center_x(), bottom};
	}

	Point left_center() const {
		return {left, center_y()};
	}

	Point right_center() const {
		return {right, center_y()};
	}
};

inline void to_json(nlohmann::json& j, const nn::Rect& rect) {
	j["left"] = rect.left;
	j["top"] = rect.top;
	j["right"] = rect.right;
	j["bottom"] = rect.bottom;
}

inline void from_json(const nlohmann::json& j, nn::Rect& rect) {
	rect.left = j.at("left");
	rect.top = j.at("top");
	rect.right = j.at("right");
	rect.bottom = j.at("bottom");
}

}