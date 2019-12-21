
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include <memory>
#include "LabelingWu.hpp"

namespace Ort {
class Env;
class Session;
class Value;
}

namespace nn {

struct TextDetector {
	TextDetector();
	~TextDetector();

	TextDetector(const TextDetector& root) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	std::vector<Rect> detect(const stb::Image& image);

private:
	void run_nn(const stb::Image& image);
	std::vector<Rect> find_rects();

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

	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};

}
