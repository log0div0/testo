
#include "Homm3Detector.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

Homm3Detector& Homm3Detector::instance() {
	static Homm3Detector instance;
	return instance;
}

std::vector<Homm3Object> Homm3Detector::detect(const stb::Image<stb::RGB>* srch_img, const std::string& id)
{
	//TODO
	return {};
}

}
