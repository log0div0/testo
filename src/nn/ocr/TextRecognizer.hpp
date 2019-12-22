
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include <string>
#include "Rect.hpp"
#include "OnnxRuntime.hpp"

namespace nn {

struct TextRecognizer {
	TextRecognizer();
	~TextRecognizer();

	TextRecognizer(const TextRecognizer& root) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	std::vector<std::string> recognize(const stb::Image& image, const std::vector<Rect>& rects);

private:
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;
};

}
