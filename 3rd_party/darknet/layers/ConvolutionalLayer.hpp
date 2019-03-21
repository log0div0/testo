
#pragma once

extern "C" {
#include <convolutional_layer.h>
}

#include <inipp.hh>

namespace darknet {

struct ConvolutionalLayer: layer {
	ConvolutionalLayer(const inipp::inisection& section,
		size_t batch,
		size_t w,
		size_t h,
		size_t c);
};

}
