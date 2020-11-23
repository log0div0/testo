
#include "ImgDetector.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

static inline bool is_n_times_div_by_2(int value, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

static inline int nearest_n_times_div_by_2(int value, size_t n) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

ImgDetector& ImgDetector::instance() {
	static ImgDetector instance;
	return instance;
}

std::vector<Img> ImgDetector::detect(const stb::Image<stb::RGB>* image, const std::string& ref_img_path)
{
	if (!image->data) {
		return {};
	}

	throw std::runtime_error("Implement ImgDetector::detect");
	// run_nn(image);
	// return run_postprocessing();
}

}
