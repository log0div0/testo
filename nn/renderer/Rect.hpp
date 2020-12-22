
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

	void shrink_left(int val) {
		x += val;
		w -= val;
	}

	void shrink_top(int val) {
		y += val;
		h -= val;
	}

	void shrink_right(int val) {
		w -= val;
	}

	void shrink_bottom(int val) {
		h -= val;
	}

	template <typename Pixel, typename Cond>
	bool extend_left(const stb::Image<Pixel>& img, Cond&& cond) {
		if (x == 0) {
			return false;
		}
		int pos_x = x - 1;
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		extend_left(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool extend_top(const stb::Image<Pixel>& img, Cond&& cond) {
		if (y == 0) {
			return false;
		}
		int pos_y = y - 1;
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		extend_top(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool extend_right(const stb::Image<Pixel>& img, Cond&& cond) {
		if (end_x() == img.w) {
			return false;
		}
		int pos_x = end_x();
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		extend_right(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool extend_bottom(const stb::Image<Pixel>& img, Cond&& cond) {
		if (end_y() == img.h) {
			return false;
		}
		int pos_y = end_y();
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		extend_bottom(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool shrink_left(const stb::Image<Pixel>& img, Cond&& cond) {
		if (w == 0) {
			return false;
		}
		int pos_x = x;
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		shrink_left(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool shrink_top(const stb::Image<Pixel>& img, Cond&& cond) {
		if (h == 0) {
			return false;
		}
		int pos_y = y;
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		shrink_top(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool shrink_right(const stb::Image<Pixel>& img, Cond&& cond) {
		if (w == 0) {
			return false;
		}
		int pos_x = end_x() - 1;
		for (int pos_y = y; pos_y < end_y(); ++pos_y) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		shrink_right(1);
		return true;
	}

	template <typename Pixel, typename Cond>
	bool shrink_bottom(const stb::Image<Pixel>& img, Cond&& cond) {
		if (h == 0) {
			return false;
		}
		int pos_y = end_y() - 1;
		for (int pos_x = x; pos_x < end_x(); ++pos_x) {
			if (!cond(img.at(pos_x, pos_y))) {
				return false;
			}
		}
		shrink_bottom(1);
		return true;
	}

	static Rect get_bitmap_bbox(const stb::Image<uint8_t>& bitmap) {
		Rect bbox(bitmap);
		while (bbox.shrink_left(bitmap, [](uint8_t x) { return x == 0; }));
		while (bbox.shrink_top(bitmap, [](uint8_t x) { return x == 0; }));
		while (bbox.shrink_right(bitmap, [](uint8_t x) { return x == 0; }));
		while (bbox.shrink_bottom(bitmap, [](uint8_t x) { return x == 0; }));
		return bbox;
	}

	static Rect get_visible_bbox(const stb::Image<stb::RGBA>& bitmap) {
		Rect bbox(bitmap);
		while (bbox.shrink_left(bitmap, [](const stb::RGBA& x) { return x.a != 255; }));
		while (bbox.shrink_top(bitmap, [](const stb::RGBA& x) { return x.a != 255; }));
		while (bbox.shrink_right(bitmap, [](const stb::RGBA& x) { return x.a != 255; }));
		while (bbox.shrink_bottom(bitmap, [](const stb::RGBA& x) { return x.a != 255; }));
		return bbox;
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