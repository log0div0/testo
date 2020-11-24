
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

	void run_nn(const stb::Image<stb::RGB>* image, const std::string& ref_img_path);
	std::vector<Img> run_postprocessing();
	std::vector<Rect> find_rects();

	LabelingWu labeling_wu;

	onnx::Model model = "ImgDetector";
	onnx::Image srch = "srch_img";
	onnx::Image ref = "ref_img";
	onnx::Image out = "output";

	std::string ref_img_path;
	int srch_w = 0, srch_h = 0;
	int ref_w = 0, ref_h = 0;
	int SRCH_W = 0, SRCH_H = 0;
	float ratio_w = 0, ratio_h = 0;
	int out_w = 0, out_h = 0;
};

}
