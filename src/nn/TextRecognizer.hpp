
#pragma once

#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct TextRecognizer {
	static TextRecognizer& instance();
	~TextRecognizer();

	TextRecognizer(const TextRecognizer&) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	void recognize(const stb::Image<stb::RGB>* image, TextLine& textline);

private:
	TextRecognizer();

	void run_nn(const stb::Image<stb::RGB>* image, TextLine& textline);
	void run_postprocessing(TextLine& textline);

	std::vector<std::vector<std::string>> symbols;
	std::vector<size_t> symbols_indexes;

	int in_w = 0;
	int in_c = 0;
	int out_w = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;

	std::vector<uint8_t> textline_img, textline_img_resized;
};

}
