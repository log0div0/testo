
#include "YoloLayer.hpp"

extern "C" {
#include <yolo_layer.h>
}

using namespace inipp;

namespace darknet {

YoloLayer::YoloLayer(const inisection& section,
	size_t batch,
	size_t w,
	size_t h,
	size_t c)
{
	int classes = section.get_int("classes", 0);
	int max_boxes = section.get_int("max_boxes");

	(layer&)*this = make_yolo_layer(batch, w, h, classes, max_boxes);

	ignore_thresh = section.get_float("ignore_thresh", .5);
	anchor_w = section.get_int("anchor_w");
	anchor_h = section.get_int("anchor_h");
}

}
