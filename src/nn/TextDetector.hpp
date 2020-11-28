
#pragma once

#include "LabelingWu.hpp"
#include "OnnxRuntime.hpp"
#include "TextLine.hpp"

namespace nn {

struct TextDetector {
	static TextDetector& instance();

	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	std::vector<TextLine> detect(const stb::Image<stb::RGB>* image);

private:
	TextDetector() = default;
	void run_nn(const stb::Image<stb::RGB>* image);
	std::vector<TextLine> run_postprocessing();
	std::vector<Rect> find_rects(int c);

	int in_w = 0;
	int in_h = 0;
	int out_w = 0;
	int out_h = 0;
	std::array<LabelingWu, 2> labeling_wu;

	onnx::Model model = "TextDetector";
	onnx::Image in = "input";
	onnx::Image out = "output";
};

}
