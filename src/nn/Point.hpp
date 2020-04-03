
#pragma once

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

}
