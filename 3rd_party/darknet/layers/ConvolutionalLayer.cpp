
#include "ConvolutionalLayer.hpp"

extern "C" {
#include <convolutional_layer.h>
}

using namespace inipp;

namespace darknet {

ConvolutionalLayer::ConvolutionalLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	int n = section.get_int("filters",1);
	int size = section.get_int("size",1);
	int stride = section.get_int("stride",1);
	int padding = section.get_int("padding",0);
	int groups = section.get_int("groups", 1);

	ACTIVATION activation = get_activation((char*)section.get("activation", "logistic").c_str());

	if(!(h && w && c)) {
		throw std::runtime_error("Layer before convolutional layer must output image.");
	}
	int batch_normalize = section.get_int("batch_normalize", 0);
	int binary = section.get_int("binary", 0);
	int xnor = section.get_int("xnor", 0);

	(layer&)*this = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor);
}

}