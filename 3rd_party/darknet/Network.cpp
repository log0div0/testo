
#include "Network.hpp"
#include <stdexcept>
#include <inipp.hh>

extern "C" {
#include <activations.h>
#include <convolutional_layer.h>
#include <batchnorm_layer.h>
#include <yolo_layer.h>
#include <maxpool_layer.h>
}

using namespace inipp;

namespace darknet {

struct size_params {
	int batch;
	int inputs;
	int h;
	int w;
	int c;
	network *net;
};

struct ConvolutionalLayer: layer {
	ConvolutionalLayer(const inisection& section, size_params params) {
		int n = section.get_int("filters",1);
		int size = section.get_int("size",1);
		int stride = section.get_int("stride",1);
		int padding = section.get_int("padding",0);
		int groups = section.get_int("groups", 1);

		ACTIVATION activation = get_activation((char*)section.get("activation", "logistic").c_str());

		int batch,h,w,c;
		h = params.h;
		w = params.w;
		c = params.c;
		batch=params.batch;
		if(!(h && w && c)) {
			throw std::runtime_error("Layer before convolutional layer must output image.");
		}
		int batch_normalize = section.get_int("batch_normalize", 0);
		int binary = section.get_int("binary", 0);
		int xnor = section.get_int("xnor", 0);

		(layer&)*this = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor);
	}
};

int *parse_yolo_mask(const char *a, int *num)
{
	int *mask = 0;
	if(a){
		int len = strlen(a);
		int n = 1;
		int i;
		for(i = 0; i < len; ++i){
			if (a[i] == ',') ++n;
		}
		mask = (int*)calloc(n, sizeof(int));
		for(i = 0; i < n; ++i){
			int val = atoi(a);
			mask[i] = val;
			a = strchr(a, ',')+1;
		}
		*num = n;
	}
	return mask;
}

struct YoloLayer: layer {
	YoloLayer(const inisection& section, size_params params)
	{
		int classes = section.get_int("classes", 0);
		int total = section.get_int("num", 1);
		int num = total;

		int *mask = parse_yolo_mask(section.get("mask").c_str(), &num);
		int max_boxes = section.get_int("max", 90);
		(layer&)*this = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
		assert(outputs == params.inputs);

		ignore_thresh = section.get_float("ignore_thresh", .5);
		truth_thresh = section.get_float("truth_thresh", 1);

		const char* a = section.get("anchors").c_str();
		if(a){
			int len = strlen(a);
			int n = 1;
			int i;
			for(i = 0; i < len; ++i){
				if (a[i] == ',') ++n;
			}
			for(i = 0; i < n; ++i){
				float bias = atof(a);
				biases[i] = bias;
				a = strchr(a, ',')+1;
			}
		}
	}
};

struct MaxPoolLayer: layer {
	MaxPoolLayer(const inisection& section, size_params params)
	{
		int stride = section.get_int("stride", 1);
		int size = section.get_int("size", stride);
		int padding = section.get_int("padding", size-1);

		int batch,h,w,c;
		h = params.h;
		w = params.w;
		c = params.c;
		batch=params.batch;
		if(!(h && w && c)) {
			throw std::runtime_error("Layer before maxpool layer must output image.");
		}

		(layer&)*this = make_maxpool_layer(batch,h,w,c,size,stride,padding);
	}
};

Network::Network(const std::string& path): network({}) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	inifile ini(file);

	n = ini.sections().size();
	layers = (layer*)calloc(n, sizeof(layer));
	seen = (size_t*)calloc(1, sizeof(size_t));
	t    = (int*)calloc(1, sizeof(int));
	cost = (float*)calloc(1, sizeof(float));
	gpu_index = gpu_index;
	batch = ini.get_int("batch", 1);
	learning_rate = ini.get_float("learning_rate", .001);
	momentum = ini.get_float("momentum", .9);
	decay = ini.get_float("decay", .0001);
	h = ini.get_int("height", 0);
	w = ini.get_int("width", 0);
	c = ini.get_int("channels", 0);
	inputs = h * w * c;

	if (!inputs && !(h && w && c)) {
		throw std::runtime_error("No input parameters supplied");
	}

	size_params params;
	params.h = h;
	params.w = w;
	params.c = c;
	params.inputs = inputs;
	params.batch = batch;
	params.net = this;

	size_t workspace_size = 0;
	int count = 0;
	fprintf(stderr, "layer     filters    size              input                output\n");
	for (auto& section: ini.sections()) {
		fprintf(stderr, "%5d ", count);

		layer l = {};
		if (section.name() == "convolutional") {
			l = ConvolutionalLayer(section, params);
		} else if (section.name() == "yolo") {
			l = YoloLayer(section, params);
		} else if (section.name() == "maxpool"){
			l = MaxPoolLayer(section, params);
		} else {
			throw std::runtime_error("Unknown layer type: " + section.name());
		}
		layers[count] = l;

		params.h = l.out_h;
		params.w = l.out_w;
		params.c = l.out_c;
		params.inputs = l.outputs;
		if (l.workspace_size > workspace_size) {
			workspace_size = l.workspace_size;
		}
		++count;
	}
	layer out = back();
	outputs = out.outputs;
	truths = out.outputs;
	if (out.truths) {
		truths = out.truths;
	}
	output = out.output;
	input = (float*)calloc(inputs*batch, sizeof(float));
	truth = (float*)calloc(truths*batch, sizeof(float));
#ifdef GPU
	output_gpu = out.output_gpu;
	input_gpu = cuda_make_array(input, inputs*batch);
	truth_gpu = cuda_make_array(truth, truths*batch);
#endif
	if(workspace_size){
#ifdef GPU
		if(gpu_index >= 0){
			workspace = (float*)cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
		}else {
			workspace = (float*)calloc(1, workspace_size);
		}
#else
		workspace = (float*)calloc(1, workspace_size);
#endif
	}
}

Network::~Network() {
	if (layers) {
		for (int i = 0; i < n; ++i) {
			free_layer(layers[i]);
		}
		free(layers);
	}
	if(input) free(input);
	if(truth) free(truth);
#ifdef GPU
	if(input_gpu) cuda_free(input_gpu);
	if(truth_gpu) cuda_free(truth_gpu);
#endif
}

Network::Network(Network&& other): network(other) {
	(network&)other = {};
}

void Network::load_weights(const std::string& weights_file_path) {
	::load_weights(this, (char*)weights_file_path.c_str());
}

void Network::save_weights(const std::string& weights_file_path) {
	::save_weights(this, (char*)weights_file_path.c_str());
}

void Network::set_batch(size_t batch) {
	set_batch_network(this, batch);
}

float* Network::predict(const Image& image) {
	if ((width() != image.width()) ||
		(height() != image.height()))
	{
		throw std::runtime_error("Image size is not equal to network size");
	} else {
		return forward(image.data);
	}
}

float* Network::forward(float* in) {
	network orig = *this;
	input = in;
	truth = 0;
	train = 0;
	delta = 0;
	forward_network(this);
	float *out = output;
	(network&)*this = orig;
	return out;
}

}
