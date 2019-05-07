
#include "YOLO.hpp"
#include "utf8.hpp"

extern "C" {
#include "src/activations.h"
}

namespace yolo {

float anchor_w = 8;
float anchor_h = 16;
std::string classes_str = R"(0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)";
std::vector<std::string> classes = utf8::split_to_chars(classes_str);

float mag_array(float *a, int n)
{
	int i;
	float sum = 0;
	for(i = 0; i < n; ++i){
		sum += a[i]*a[i];
	}
	return sqrt(sum);
}

float delta_yolo_box(darknet::Layer& l, const Box& truth, int index, int i, int j, int w, int h)
{
	float scale = 2-truth.w*truth.h;
	int stride = l.out_w*l.out_h;

	Box pred = {};
	pred.x = (i + logistic_activate(l.output[index + 0*stride])) / l.out_w;
	pred.y = (j + logistic_activate(l.output[index + 1*stride])) / l.out_h;
	pred.w = exp(l.output[index + 2*stride]) * anchor_w / w;
	pred.h = exp(l.output[index + 3*stride]) * anchor_h / h;

	float iou = pred.uio(truth);

	float tx = (truth.x*l.out_w - i);
	float ty = (truth.y*l.out_h - j);
	float tw = log(truth.w*w / anchor_w);
	float th = log(truth.h*h / anchor_h);

	l.delta[index + 0*stride] = scale * (tx - logistic_activate(l.output[index + 0*stride]));
	l.delta[index + 1*stride] = scale * (ty - logistic_activate(l.output[index + 1*stride]));
	l.delta[index + 2*stride] = scale * (tw - l.output[index + 2*stride]);
	l.delta[index + 3*stride] = scale * (th - l.output[index + 3*stride]);
	return iou;
}


float delta_yolo_class(darknet::Layer& l, int index, int class_)
{
	float result = 0;
	int stride = l.out_w*l.out_h;
	for(size_t n = 0; n < classes.size(); ++n){
		l.delta[index + stride*n] = ((n == (size_t)class_)?1 : 0) - logistic_activate(l.output[index + stride*n]);
		if (n == (size_t)class_) {
			result += logistic_activate(l.output[index + stride*n]);
		}
	}
	return result;
}

int entry_index(darknet::Layer& l, int batch, int location, int entry)
{
	return batch*l.outputs + entry*l.out_w*l.out_h + location;
}

float train(darknet::Network& network, Dataset& dataset,
	float learning_rate,
	float momentum,
	float decay)
{
	auto labels = dataset.charge(&network);
	network.train = true;
	network.forward();

	float loss = 0;
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	int count = 0;
	auto& l = *network.layers.back();
	for (int b = 0; b < l.batch; ++b) {
		auto& label = labels.at(b);
		for (int j = 0; j < l.out_h; ++j) {
			for (int i = 0; i < l.out_w; ++i) {
				int obj_index = entry_index(l, b, j*l.out_w + i, 4);
				l.delta[obj_index] = 0 - logistic_activate(l.output[obj_index]);
			}
		}
		for(auto& truth: label){

			int i = (truth.x * l.out_w);
			int j = (truth.y * l.out_h);

			int box_index = entry_index(l, b, j*l.out_w + i, 0);
			float iou = delta_yolo_box(l, truth, box_index, i, j, network.w, network.h);

			int obj_index = entry_index(l, b, j*l.out_w + i, 4);
			avg_obj += logistic_activate(l.output[obj_index]);
			l.delta[obj_index] = 1 - logistic_activate(l.output[obj_index]);

			int class_index = entry_index(l, b, j*l.out_w + i, 4 + 1);
			avg_cat += delta_yolo_class(l, class_index, truth.class_id);

			++count;
			if(iou > .5) recall += 1;
			if(iou > .75) recall75 += 1;
			avg_iou += iou;
		}
	}
	loss = pow(mag_array(l.delta, l.outputs * l.batch), 2) / l.batch;
	printf("Avg IOU: %f, Class: %f, Obj: %f, .5R: %f, .75R: %f,  count: %d\n", avg_iou/count, avg_cat/count, avg_obj/count, recall/count, recall75/count, count);

	network.backward();
	network.update(learning_rate, momentum, decay);

	return loss;
}

bool find_substr(const stb::Image& image, const darknet::Layer& l,
	int left, int top,
	const std::vector<std::string>& query, size_t index,
	std::vector<Rect>& rects
) {
	int right = left;
	int bottom = top;
	while (true) {
		if (index == query.size()) {
			return true;
		}
		right += 3;
		bottom += 2;
		if (query.at(index) != " ") {
			break;
		} else {
			++index;
		}
	}
	size_t dimension_size = l.out_w * l.out_h;
	auto it = std::find(classes.begin(), classes.end(), query.at(index));
	if (it == classes.end()) {
		throw std::runtime_error("Unsupported symbol: " + query.at(index));
	}
	size_t class_id = it - classes.begin();
	for (int x = left; (x < right) && (x < l.out_w); ++x) {
		for (int y = top; (y < bottom) && (y < l.out_h); ++y) {
			int i = y * l.out_w + x;

			float objectness = logistic_activate(l.output[dimension_size * 4 + i]);
			if (objectness < 0.1f) {
				continue;
			}

			float class_probability = logistic_activate(l.output[dimension_size * (5 + class_id) + i]);
			if (class_probability < 0.5f) {
				continue;
			}

			Box b;
			b.x = (x + logistic_activate(l.output[dimension_size * 0 + i])) / l.out_w;
			b.y = (y + logistic_activate(l.output[dimension_size * 1 + i])) / l.out_h;
			b.w = exp(l.output[dimension_size * 2 + i]) * anchor_w / image.width;
			b.h = exp(l.output[dimension_size * 3 + i]) * anchor_h / image.height;

			Rect rect;
			rect.left = (b.x-b.w/2)*image.width;
			rect.right = (b.x+b.w/2)*image.width;
			rect.top = (b.y-b.h/2)*image.height;
			rect.bottom = (b.y+b.h/2)*image.height;

			rects.push_back(rect);

			return find_substr(image, l, x + 1, top, query, index + 1, rects);
		}
	}
	return false;
}

bool predict(darknet::Network& network, stb::Image& image, const std::string& text) {

	bool result = false;

	network.train = false;

	uint8_t buffer[sizeof(darknet::Image)];
	darknet::Image& tmp = *(darknet::Image*)buffer;
	tmp.data = network.input;
	tmp.w = network.w;
	tmp.h = network.h;
	tmp.c = network.c;
	tmp.from_stb(image);

	network.forward();

	std::vector<std::string> query = utf8::split_to_chars(text);

	const darknet::Layer& l = *network.layers.back();

	for (int y = 0; y < l.out_h; ++y) {
		for (int x = 0; x < l.out_w; ++x) {
			std::vector<Rect> rects;
			if (find_substr(image, l, x, y, query, 0, rects)) {
				result = true;
				for (auto& rect: rects) {
					image.draw(rect.left, rect.top, rect.right, rect.bottom, 200, 20, 50);
				}
			}

		}
	}
	return result;
}

}