
#pragma once

#include <nlohmann/json.hpp>
#include <cstdint>

namespace nn {

struct Point {
	int32_t x = 0, y = 0;

	Point move_up(int32_t v) const {
		return {x, y - v};
	}

	Point move_down(int32_t v) const {
		return {x, y + v};
	}

	Point move_left(int32_t v) const {
		return {x - v, y};
	}

	Point move_right(int32_t v) const {
		return {x + v, y};
	}
};

inline void to_json(nlohmann::json& j, const nn::Point& point) {
	j["type"] = "Point";
	j["x"] = point.x;
	j["y"] = point.y;
}

inline void from_json(const nlohmann::json& j, nn::Point& point) {
	point.x = j.at("x");
	point.y = j.at("y");
}

}
