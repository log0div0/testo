
#include "MaxPoolLayer.hpp"

extern "C" {
#include <maxpool_layer.h>
}

using namespace inipp;

namespace darknet {

MaxPoolLayer::MaxPoolLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	int stride = section.get_int("stride", 1);
	int size = section.get_int("size", stride);
	int padding = section.get_int("padding", size-1);

	if(!(h && w && c)) {
		throw std::runtime_error("Layer before maxpool layer must output image.");
	}

	(layer&)*this = make_maxpool_layer(batch,h,w,c,size,stride,padding);
}

}
