
#pragma once

#include <vector>
#include <stb/Image.hpp>
#include "Homm3Object.hpp"

namespace nn {

struct Homm3Detector {
	static Homm3Detector& instance();

	Homm3Detector(const Homm3Detector& root) = delete;
	Homm3Detector& operator=(const Homm3Detector&) = delete;

	std::vector<Homm3Object> detect(const stb::Image<stb::RGB>* image, const std::string& id);

private:
	Homm3Detector() = default;
};

}
