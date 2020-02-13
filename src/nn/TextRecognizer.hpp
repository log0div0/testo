
#pragma once

#include "OnnxRuntime.hpp"
#include "OCR.hpp"

namespace nn {

struct TextRecognizer {
	static TextRecognizer& instance();
	~TextRecognizer();

	TextRecognizer(const TextRecognizer&) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	std::vector<Char> recognize(const Word& word);

private:
	TextRecognizer();

	void run_nn(const Word& word);
	std::vector<Char> run_postprocessing(const Word& word);

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

	std::vector<uint8_t> word_img, word_img_resized;
};

}