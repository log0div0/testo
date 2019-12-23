
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include "Word.hpp"
#include "OnnxRuntime.hpp"

namespace nn {

struct TextRecognizer {
	TextRecognizer();
	~TextRecognizer();

	TextRecognizer(const TextRecognizer& root) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	void recognize(const stb::Image& image, std::vector<Word>& words);

private:
	void run_nn(const stb::Image& image, const std::vector<Word>& words);
	void decode_words(std::vector<Word>& words);

	std::vector<std::string> symbols;

	int batch_size = 0;
	int in_w = 0;
	int in_c = 0;
	int out_w = 0;
	int out_c = 0;
	std::vector<float> in;
	std::vector<float> out;

	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Value> in_tensor;
	std::unique_ptr<Ort::Value> out_tensor;

	std::vector<int> widths;
	std::vector<uint8_t> word_grey, word_grey_resized;
};

}
