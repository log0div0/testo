
#pragma once

#include <darknet.h>
#include <inipp.hh>

namespace darknet {

struct MaxPoolLayer: layer {
	MaxPoolLayer(const inipp::inisection& section,
		size_t batch,
		size_t w,
		size_t h,
		size_t c);
};

}
