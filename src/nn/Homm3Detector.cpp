
#include "Homm3Detector.hpp"
#include <iostream>
#include <algorithm>

namespace nn {

Homm3Detector& Homm3Detector::instance() {
	static Homm3Detector instance;
	return instance;
}

}
