
#pragma once

#include <stb/Image.hpp>
#include <nlohmann/json.hpp>

struct Rect {
	int32_t x = 0, y = 0, w = 0, h = 0;

	Rect() = default;
	template <typename T>
	Rect(const stb::Image<T>& img) {
		x = 0;
		y = 0;
		w = img.w;
		h = img.h;
	}

	int32_t area() const {
		return w * h;
	}

	Rect operator|(const Rect& other) const {
		Rect result;
		result.x = std::min(x, other.x);
		result.y = std::min(y, other.y);
		result.w = std::max(end_x(), other.end_x()) - result.x;
		result.h = std::max(end_y(), other.end_y()) - result.y;
		return result;
	}

	Rect operator&(const Rect& other) const {
		if (x >= other.end_x()) {
			return {};
		}
		if (y >= other.end_y()) {
			return {};
		}
		if (end_x() <= other.x) {
			return {};
		}
		if (end_y() <= other.y) {
			return {};
		}
		Rect result;
		result.x = std::max(x, other.x);
		result.y = std::max(y, other.y);
		result.w = std::min(end_x(), other.end_x()) - result.x;
		result.h = std::min(end_y(), other.end_y()) - result.y;
		return result;
	}

	int32_t end_x() const {
		return x + w;
	}

	int32_t end_y() const {
		return y + h;
	}

	void extend(int val) {
		x -= val;
		y -= val;
		h += val * 2;
		w += val * 2;
	}

	void extend_left(int val) {
		x -= val;
		w += val;
	}

	void extend_top(int val) {
		y -= val;
		h += val;
	}

	void extend_right(int val) {
		w += val;
	}

	void extend_bottom(int val) {
		h += val;
	}

	void shrink_top(int val) {
		y += val;
		h -= val;
	}

	void shrink_bottom(int val) {
		h -= val;
	}

	bool extend_left(const stb::Image<stb::RGB>& img, const stb::RGB& color) {
		if (x == 0) {
			return false;
		}
		int pos_x = x - 1;
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (img.at(pos_x, pos_y).max_channel_diff(color) > 32) {
				return false;
			}
		}
		--x;
		++w;
		return true;
	}

	bool extend_top(const stb::Image<stb::RGB>& img, const stb::RGB& color) {
		if (y == 0) {
			return false;
		}
		int pos_y = y - 1;
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (img.at(pos_x, pos_y).max_channel_diff(color) > 32) {
				return false;
			}
		}
		--y;
		++h;
		return true;
	}

	bool extend_right(const stb::Image<stb::RGB>& img, const stb::RGB& color) {
		if (end_x() == img.w) {
			return false;
		}
		int pos_x = end_x();
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (img.at(pos_x, pos_y).max_channel_diff(color) > 32) {
				return false;
			}
		}
		++w;
		return true;
	}

	bool extend_bottom(const stb::Image<stb::RGB>& img, const stb::RGB& color) {
		if (end_y() == img.h) {
			return false;
		}
		int pos_y = end_y();
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (img.at(pos_x, pos_y).max_channel_diff(color) > 32) {
				return false;
			}
		}
		++h;
		return true;
	}
};

void from_json(const nlohmann::json& j, Rect& rect) {
	rect.x = floorf(j.at("x").get<float>());
	rect.y = floorf(j.at("y").get<float>());
	rect.w = ceilf(j.at("width").get<float>() + (j.at("x").get<float>() - floorf(j.at("x").get<float>())));
	rect.h = ceilf(j.at("height").get<float>() + (j.at("y").get<float>() - floorf(j.at("y").get<float>())));
}

void to_json(nlohmann::json& j, const Rect& rect) {
	j["x"] = rect.x;
	j["y"] = rect.y;
	j["width"] = rect.w;
	j["height"] = rect.h;
}