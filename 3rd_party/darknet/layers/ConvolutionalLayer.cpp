
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

	ACTIVATION activation = get_activation((char*)section.get("activation", "logistic").c_str());

	if(!(h && w && c)) {
		throw std::runtime_error("Layer before convolutional layer must output image.");
	}
	int batch_normalize = section.get_int("batch_normalize", 0);

	(layer&)*this = make_convolutional_layer(batch,h,w,c,n,size,stride,padding,activation, batch_normalize);
}

void ConvolutionalLayer::load_weights(FILE* fp)
{
    int num = c*n*size*size;
    fread(biases, sizeof(float), n, fp);
    if (batch_normalize){
        fread(scales, sizeof(float), n, fp);
        fread(rolling_mean, sizeof(float), n, fp);
        fread(rolling_variance, sizeof(float), n, fp);
    }
    fread(weights, sizeof(float), num, fp);
#ifdef GPU
    if(use_gpu){
        push_convolutional_layer(*this);
    }
#endif
}

void ConvolutionalLayer::save_weights(FILE* fp) const
{
#ifdef GPU
	if(use_gpu){
		pull_convolutional_layer(*this);
	}
#endif
	int num = nweights;
	fwrite(biases, sizeof(float), n, fp);
	if (batch_normalize){
		fwrite(scales, sizeof(float), n, fp);
		fwrite(rolling_mean, sizeof(float), n, fp);
		fwrite(rolling_variance, sizeof(float), n, fp);
	}
	fwrite(weights, sizeof(float), num, fp);
}

}
