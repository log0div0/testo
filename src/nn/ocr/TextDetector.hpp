
#pragma once

#include <stb/Image.hpp>
#include "LabelingWu.hpp"
#include "OnnxRuntime.hpp"

namespace nn {

struct TextDetector {
	TextDetector();
	~TextDetector();

	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	std::vector<Rect> detect(const stb::Image& image);

private:
	void run_nn(const stb::Image& image);
	std::vector<Rect> find_words();
	std::vector<Rect> find_chars();
	Rect adjust_rect(const Rect& rect, float threshold);

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
	LabelingWu labelingWu;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};

}
