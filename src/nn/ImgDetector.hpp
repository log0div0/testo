
#pragma once

#include "LabelingWu.hpp"
#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct ImgDetector {
	static ImgDetector& instance();
	~ImgDetector();

	ImgDetector(const ImgDetector& root) = delete;
	ImgDetector& operator=(const ImgDetector&) = delete;

	std::vector<Img> detect(const stb::Image<stb::RGB>* image, const std::string& ref_img_path);

private:
	ImgDetector();
	void run_nn(const stb::Image<stb::RGB>* image);
	std::vector<Img> run_postprocessing();
	std::vector<Rect> find_rects();

	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;
	LabelingWu labeling_wu;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};

}
