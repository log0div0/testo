
#include "Homm3Tensor.hpp"
#include "Homm3Detector.hpp"

namespace nn {

Homm3Tensor find_homm3(const stb::Image<stb::RGB>* image) {
	Homm3Tensor result;

	for (auto obj: Homm3Detector::instance().detect(image)) {
		if (obj.class_name == "hovel") {
			obj.rect.top += obj.rect.height() / 2;
			obj.rect.left += obj.rect.width() / 2;
		} else if (obj.class_name == "redwood_observatory") {
			obj.rect.top += obj.rect.height() / 3 * 2;
		} else if (obj.class_name == "magic_spring") {
			obj.rect.top += obj.rect.height() / 2;
		}
		result.objects.push_back(obj);
	}

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
