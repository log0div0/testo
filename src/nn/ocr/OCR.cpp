
#include "OCR.hpp"

namespace nn {

std::vector<TextLine> OCR::run(const stb::Image& image) {
	std::vector<Word> words = detector.detect(image);
	recognizer.recognize(image, words);

	throw std::runtime_error("Impelement me");
}

}
