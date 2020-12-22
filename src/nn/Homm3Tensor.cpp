
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

Homm3Tensor Homm3Tensor::match_class(const stb::Image<stb::RGB>* image, const std::string& class_name) {
	Homm3Tensor result;
	for (auto& obj: objects) {
		if (obj.class_name == class_name) {
			result.objects.push_back(obj);
		}
	}
	return result;
}

}
