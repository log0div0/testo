
#pragma once

#include <stb/Image.hpp>
#include <vector>
#include "TextLine.hpp"
#include "OnnxRuntime.hpp"

namespace nn {

struct TextRecognizer {
	TextRecognizer();
	~TextRecognizer();

	TextRecognizer(const TextRecognizer& root) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;

	std::vector<Char> recognize(const stb::Image& image, Word& word);

private:
	void run_nn(const stb::Image& image, const Word& word);
	std::vector<Char> decode_word(Word& word);

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

	std::vector<uint8_t> word_grey, word_grey_resized;
};

}
