
#pragma once

#include "OCR.hpp"
#include <memory>

namespace nn {

struct Context {
	Context(const stb::Image* image_);

	OCR& ocr();

private:
	const stb::Image* image = nullptr;
	std::unique_ptr<OCR> _ocr;
};

}
