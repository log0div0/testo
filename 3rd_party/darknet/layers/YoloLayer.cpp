
#include "YoloLayer.hpp"

using namespace inipp;

namespace darknet {

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

YoloLayer::YoloLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	int classes = section.get_int("classes", 0);
	int total = section.get_int("num", 1);
	int num = total;

	int *mask = parse_yolo_mask(section.get("mask").c_str(), &num);
	int max_boxes = section.get_int("max", 90);
	(layer&)*this = make_yolo_layer(batch, w, h, num, total, mask, classes, max_boxes);
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

}
