
#pragma once

#include "LabelingWu.hpp"
#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct ImgDetector {
	static ImgDetector& instance();

	ImgDetector(const ImgDetector& root) = delete;
	ImgDetector& operator=(const ImgDetector&) = delete;

	std::vector<Img> detect(const stb::Image<stb::RGB>* image, const std::string& ref_img_path);

private:
	ImgDetector() = default;

	void run_nn(const stb::Image<stb::RGB>* image);
	std::vector<Img> run_postprocessing();
	std::vector<Rect> find_rects();

	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	LabelingWu labeling_wu;

	onnx::Model model = "ImgDetector";
	onnx::Image in = "input";
	onnx::Image out = "output";
};

}
