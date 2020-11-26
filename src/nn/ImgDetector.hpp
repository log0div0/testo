
#pragma once

#include "OCR.hpp"

namespace nn {

struct ImgDetector {
	static ImgDetector& instance();

	ImgDetector(const ImgDetector& root) = delete;
	ImgDetector& operator=(const ImgDetector&) = delete;

	std::vector<Img> detect(const stb::Image<stb::RGB>* image, const std::string& ref_img_path);

private:
	ImgDetector() = default;
};

}
