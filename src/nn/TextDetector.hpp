
#pragma once

#include "LabelingWu.hpp"
#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct TextDetector {
	static TextDetector& instance();
	~TextDetector();

	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	std::vector<TextLine> detect(const stb::Image<stb::RGB>* image);

private:
	TextDetector();
	void run_nn(const stb::Image<stb::RGB>* image);
	std::vector<TextLine> run_postprocessing();
	std::vector<Rect> find_rects(int c);

	int in_w = 0;
	int in_h = 0;
	int in_c = 0;
	int out_w = 0;
	int out_h = 0;
	int out_c = 0;
	int in_pad_w = 0;
	int in_pad_h = 0;
	int out_pad_w = 0;
	int out_pad_h = 0;
	std::vector<float> in;
	std::vector<float> out;
	std::array<LabelingWu, 2> labeling_wu;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};

}
