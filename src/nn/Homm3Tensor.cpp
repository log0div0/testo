
#include "Homm3Tensor.hpp"
#include "Homm3Detector.hpp"

namespace nn {

Homm3Tensor find_homm3(const stb::Image<stb::RGB>* image) {
	Homm3Tensor result;

	result.objects = Homm3Detector::instance().detect(image);

	std::sort(result.objects.begin(), result.objects.end(), [](const Homm3Object& a, const Homm3Object& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

}
