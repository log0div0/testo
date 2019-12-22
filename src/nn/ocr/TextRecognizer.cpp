
#include "TextRecognizer.hpp"
#include <iostream>

extern unsigned char TextRecognizer_onnx[];
extern unsigned int TextRecognizer_onnx_len;

namespace nn {

TextRecognizer::TextRecognizer() {
	session = LoadModel(TextRecognizer_onnx, TextRecognizer_onnx_len);
}

TextRecognizer::~TextRecognizer() {

}

std::vector<std::string> TextRecognizer::recognize(const stb::Image& image, const std::vector<Rect>& rects) {
	throw std::runtime_error("Implement me");
}

}
