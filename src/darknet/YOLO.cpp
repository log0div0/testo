
#include "YOLO.hpp"
#include "utf8.hpp"

extern "C" {
#include "src/activations.h"
}

namespace yolo {

float anchor_w = 8;
float anchor_h = 16;

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

float delta_yolo_class(darknet::Layer& l, int index, int classes_count, int class_id)
{
	float result = 0;
	int stride = l.out_w*l.out_h;
	if (class_id >= classes_count) {
		throw std::runtime_error("class_id >= classes_count");
	}
	for (int n = 0; n < classes_count; ++n){
		l.delta[index + stride*n] = ((n == class_id)?1 : 0) - logistic_activate(l.output[index + stride*n]);
		if (n == class_id) {
			result += logistic_activate(l.output[index + stride*n]);
		}
	}
	return result;
}

int entry_index(darknet::Layer& l, int batch, int location, int entry)
{
	return batch*l.outputs + entry*l.out_w*l.out_h + location;
}

constexpr int classes_count = 74;
constexpr int colors_count = 10;

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
	float avg_color = 0;
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
			avg_cat += delta_yolo_class(l, class_index, classes_count, truth.class_id);

			int foreground_index = entry_index(l, b, j*l.out_w + i, 4 + 1 + classes_count);
			avg_color += delta_yolo_class(l, foreground_index, colors_count, truth.foreground_id);

			int background_index = entry_index(l, b, j*l.out_w + i, 4 + 1 + classes_count + colors_count);
			avg_color += delta_yolo_class(l, background_index, colors_count, truth.background_id);

			++count;
			if(iou > .5) recall += 1;
			if(iou > .75) recall75 += 1;
			avg_iou += iou;
		}
	}
	loss = pow(mag_array(l.delta, l.outputs * l.batch), 2) / l.batch;
	printf("Avg IOU: %f, Class: %f, Color: %f, Obj: %f, .5R: %f, .75R: %f,  count: %d\n", avg_iou/count, avg_cat/count, avg_color/count/2, avg_obj/count, recall/count, recall75/count, count);

	network.backward();
	network.update(learning_rate, momentum, decay);

	return loss;
}

bool find_substr(const stb::Image& image, const darknet::Layer& l,
	int left, int top,
	const std::vector<std::string>& query,
	int foreground_id, int background_id,
	size_t index,
	const std::map<std::string, int>& symbols,
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

	int class_id = symbols.at(query.at(index));
	for (int y = top; (y < bottom) && (y < l.out_h); ++y) {
		for (int x = left; (x < right) && (x < l.out_w); ++x) {
			int i = y * l.out_w + x;

			float objectness = logistic_activate(l.output[dimension_size * 4 + i]);
			if (objectness < 0.01f) {
				continue;
			}

			// std::vector<int> v;
			// int classes_count = l.out_c - 4 - 1;
			// for (int i = 0; i < classes_count; ++i) {
			// 	v.push_back(i);
			// }
			// std::sort(v.begin(), v.end(), [&](int a, int b) {
			// 	float a_probability = logistic_activate(l.output[dimension_size * (5 + a) + i]);
			// 	float b_probability = logistic_activate(l.output[dimension_size * (5 + b) + i]);
			// 	return a_probability > b_probability;
			// });
			// auto it = std::find(v.begin(), v.end(), class_id);
			// if ((it - v.begin()) > 5) {
			// 	continue;
			// }

			float class_probability = logistic_activate(l.output[dimension_size * (5 + class_id) + i]);
			if (class_probability < 0.01f) {
				continue;
			}

			if (foreground_id >= 0) {
				float foreground_probability = logistic_activate(l.output[dimension_size * (5 + classes_count + foreground_id) + i]);
				if (foreground_probability < 0.01f) {
					continue;
				}
			}

			if (background_id >= 0) {
				float background_probability = logistic_activate(l.output[dimension_size * (5 + classes_count + colors_count + background_id) + i]);
				if (background_probability < 0.01f) {
					continue;
				}
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

			if (rects.size()) {
				if (rects.back().iou(rect) >= 0.5f) {
					continue;
				}
			}

			rects.push_back(rect);

			return find_substr(image, l, x + 1, top, query, foreground_id, background_id, index + 1, symbols, rects);
		}
	}
	return false;
}

std::map<std::string, int> load_symbols(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	return load_symbols(file);
}

std::map<std::string, int> load_symbols(std::istream& stream) {
	nlohmann::json json;
	stream >> json;
	std::map<std::string, int> result;
	for (size_t i = 0; i < json.size(); ++i) {
		std::string chars = json[i];
		for (const auto& ch: utf8::split_to_chars(chars)) {
			result[ch] = i;
		}
	}
	return result;
}

std::vector<std::string> colors = {
	"white",
	"gray",
	"black",
	"red",
	"orange",
	"yellow",
	"green",
	"cyan",
	"blue",
	"purple",
};

int get_color_id(const std::string& color) {
	if (!color.size()) {
		return -1;
	}
	for (size_t i = 0; i < colors.size(); ++i) {
		if (colors[i] == color) {
			return i;
		}
	}
	throw std::runtime_error("Unknown color \"" + color + "\"");
}

bool predict(darknet::Network& network, stb::Image& image, const std::string& text,
	const std::string& foreground,
	const std::string& background,
	const std::map<std::string, int>& symbols) {

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
	int foreground_id = get_color_id(foreground);
	int background_id = get_color_id(background);

	const darknet::Layer& l = *network.layers.back();

	for (int y = 0; y < l.out_h; ++y) {
		for (int x = 0; x < l.out_w; ++x) {
			std::vector<Rect> rects;
			if (find_substr(image, l, x, y, query, foreground_id, background_id, 0, symbols, rects)) {
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
