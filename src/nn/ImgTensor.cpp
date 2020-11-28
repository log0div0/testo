
#include "ImgTensor.hpp"
#include "ImgDetector.hpp"

namespace nn {

ImgTensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img) {
	ImgTensor result;

	result.objects = ImgDetector::instance().detect(image, path_to_img.generic_string());

	std::sort(result.objects.begin(), result.objects.end(), [](const Img& a, const Img& b) {
		return a.rect.top < b.rect.top;
	});

	return result;
}

}
