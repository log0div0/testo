
#include "Context.hpp"

namespace nn {

Context::Context(const stb::Image* image_): image(image_) {

}

OCR& Context::ocr() {
	if (!_ocr) {
		_ocr.reset(new OCR(image));
	}
	return *_ocr;
}

}
