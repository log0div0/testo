#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <clipp.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <list>
#include <fstream>
#include <inipp.hh>
#include <signal.h>
#include "Network.hpp"

extern "C" {
#include "src/activations.h"
}

using namespace darknet;


struct Box {
	float x, y, h, w;
	float uio(const Box& other) const {
		return intersection(other)/union_(other);
	}
	float intersection(const Box& other) const {
		float w = overlap(this->x, this->w, other.x, other.w);
		float h = overlap(this->y, this->h, other.y, other.h);
		if(w < 0 || h < 0) {
			return 0;
		}
		float area = w*h;
		return area;
	}
	float union_(const Box& other) const {
		float i = intersection(other);
		float u = w*h + other.w*other.h - i;
		return u;
	}

	static float overlap(float x1, float w1, float x2, float w2)
	{
		float l1 = x1 - w1/2;
		float l2 = x2 - w2/2;
		float left = l1 > l2 ? l1 : l2;
		float r1 = x1 + w1/2;
		float r2 = x2 + w2/2;
		float right = r1 < r2 ? r1 : r2;
		return right - left;
	}

};

struct Dataset {
	Dataset(const std::string& path)  {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + path);
		}
		inipp::inifile ini(file);

		item_count = std::stoi(ini.get("item_count"));

		image_width = std::stoi(ini.get("image_width"));
		image_height = std::stoi(ini.get("image_height"));
		image_channels = 3;
		image_size = image_width * image_height * image_channels;

		image_dir = ini.get("image_dir") + "/";
		label_dir = ini.get("label_dir") + "/";
	}

	struct Object: Box {
		int class_id;

		friend std::istream& operator>>(std::istream& stream, Object& object) {
			return stream
				>> object.class_id
				>> object.x
				>> object.y
				>> object.w
				>> object.h;
		}
	};


	struct Label: std::vector<Object> {
		Label(const std::string& path) {
			std::ifstream file(path);
			if (!file.is_open()) {
				throw std::runtime_error("Failed to open file " + path);
			}
			Object object;
			while (file >> object) {
				push_back(object);
			}
		}
	};

	std::vector<Label> charge(Network* network) {
		std::vector<Label> result;

		for (size_t row_index = 0; row_index < network->batch; ++row_index)
		{
			size_t item_index = rand() % item_count;

			std::string image_path = image_dir + std::to_string(item_index) + ".png";
			Image image(image_path);
			if ((image.w != image_width) ||
				(image.h != image_height) ||
				(image.c != image_channels)) {
				throw std::runtime_error("Image of invalid size");
			}
			memcpy(&network->input[image_size*row_index], image.data, image_size*sizeof(float));

			std::string label_path = label_dir + std::to_string(item_index) + ".txt";
			Label label(label_path);
			result.push_back(std::move(label));
		}

		return result;
	}

	size_t item_count;
	size_t image_size;
	size_t image_width, image_height, image_channels;
	std::string image_dir, label_dir;
};

std::string network_file;
std::string dataset_file;
std::string weights_file;
std::string image_file;
std::string output_file;
int batch_size = 32;
float learning_rate = 0.0001;
float momentum = 0.9;
float decay = 0.0005;
float thresh = 0.5f;
#ifdef GPU
int gpu = 0;
#endif

float mag_array(float *a, int n)
{
	int i;
	float sum = 0;
	for(i = 0; i < n; ++i){
		sum += a[i]*a[i];
	}
	return sqrt(sum);
}

float anchor_w = 8;
float anchor_h = 16;
std::string classes = R"(0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)";

float delta_yolo_box(Layer& l, const Box& truth, int index, int i, int j, int w, int h)
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


float delta_yolo_class(Layer& l, int index, int class_)
{
	float result = 0;
	int stride = l.out_w*l.out_h;
	for(int n = 0; n < classes.size(); ++n){
		l.delta[index + stride*n] = ((n == class_)?1 : 0) - logistic_activate(l.output[index + stride*n]);
		if (n == class_) {
			result += logistic_activate(l.output[index + stride*n]);
		}
	}
	return result;
}

int entry_index(Layer& l, int batch, int location, int entry)
{
	return batch*l.outputs + entry*l.out_w*l.out_h + location;
}

bool stop_training = false;

void sig_handler(int signum)
{
	stop_training = true;
}


void train()
{
	signal(SIGINT, sig_handler);

	Dataset dataset(dataset_file);

	Network network(network_file, batch_size, dataset.image_width, dataset.image_height, dataset.image_channels);
	if (weights_file.size()) {
		network.load_weights(weights_file);
	}
	network.train = 1;

	float avg_loss = -1;

	for (size_t i = 0; !stop_training; ++i)
	{
		auto labels = dataset.charge(&network);
		network.forward();

		float loss = 0;

		{
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
		}

		network.backward();
		network.update(learning_rate, momentum, decay);

		if (avg_loss < 0) {
			avg_loss = loss;
		} else {
			avg_loss = avg_loss*.9 + loss*.1;
		}

		std::cout << i << ": loss = " << loss << ", avg_loss = " << avg_loss << std::endl;
	}

	network.save_weights(output_file);
}

struct Rect {
	int32_t left = 0, top = 0, right = 0, bottom = 0;

	float iou(const Rect& other) const {
		return float((*this & other).area()) / (*this | other).area();
	}

	int32_t area() const {
		return (right - left) * (bottom - top);
	}

	Rect operator|(const Rect& other) const {
		return {
			std::min(left, other.left),
			std::min(top, other.top),
			std::max(right, other.right),
			std::max(bottom, other.bottom)
		};
	}
	Rect& operator|=(const Rect& other) {
		left = std::min(left, other.left);
		top = std::min(top, other.top);
		right = std::max(right, other.right);
		bottom = std::max(bottom, other.bottom);
		return *this;
	}
	Rect operator&(const Rect& other) const {
		if (left > other.right) {
			return {};
		}
		if (top > other.bottom) {
			return {};
		}
		if (right < other.left) {
			return {};
		}
		if (bottom < other.top) {
			return {};
		}
		return {
			std::max(left, other.left),
			std::max(top, other.top),
			std::min(right, other.right),
			std::min(bottom, other.bottom)
		};
	}
	Rect& operator&=(const Rect& other);

	int32_t width() const {
		return right - left;
	}

	int32_t height() const {
		return bottom - top;
	}
};

struct RectSet: std::list<Rect> {
	void add(const Rect& rect)
	{
		Rect rect_ext = rect;
		if (rect_ext.left >= 16) {
			rect_ext.left -= 16;
		} else {
			rect_ext.left = 0;
		}
		for (auto it = begin(); it != end(); ++it) {
			Rect intersectoin = *it & rect_ext;
			if (intersectoin.area()) {
				if ((intersectoin.height() * 2 > rect.height()) || (intersectoin.height() * 2 > it->height())) {
					Rect union_ = *it | rect;
					erase(it);
					add(union_);
					return;

				}
			}
		}

		push_back(rect);
	}
};

void predict()
{
	Image image = Image(image_file);

	Network network(network_file, 1, image.w, image.h, image.c);
	network.load_weights(weights_file);
	network.train = 0;

	auto start = std::chrono::high_resolution_clock::now();

	if ((image.w * image.h * image.c) != network.inputs)
	{
		throw std::runtime_error("Image size is not equal to network size");
	}

	memcpy(network.input, image.data, network.inputs * sizeof(float));
	network.delta = 0;
	network.forward();
	float* predictions = network.layers.back()->output;

	// const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";

	const auto& l = *network.layers.back();

	size_t dimension_size = l.out_w * l.out_h;

	RectSet rects;
	for (int y = 0; y < l.out_h; ++y) {
		for (int x = 0; x < l.out_w; ++x) {
			int i = y * l.out_w + x;
			float objectness = logistic_activate(predictions[dimension_size * 4 + i]);
			if (objectness < thresh) {
				continue;
			}

			Box b;
			b.x = (x + logistic_activate(predictions[dimension_size * 0 + i])) / l.out_w;
			b.y = (y + logistic_activate(predictions[dimension_size * 1 + i])) / l.out_h;
			b.w = exp(predictions[dimension_size * 2 + i]) * anchor_w / image.width();
			b.h = exp(predictions[dimension_size * 3 + i]) * anchor_h / image.height();


			Rect rect;
			rect.left = (b.x-b.w/2)*image.width();
			rect.right = (b.x+b.w/2)*image.width();
			rect.top = (b.y-b.h/2)*image.height();
			rect.bottom = (b.y+b.h/2)*image.height();

			rects.add(rect);
			image.draw(rect.left, rect.top, rect.right, rect.bottom, 0.9f, 0.2f, 0.3f);
		}
	}

	for (auto& rect: rects) {
		if (rect.height() >= 8 && rect.height() <= 24) {
			if (rect.width() >= 8) {
				// image.draw(rect.left, rect.top, rect.right, rect.bottom, 0.9f, 0.2f, 0.3f);
			}
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << time.count() << " seconds" << std::endl;

	image.save(output_file);
}

enum Mode {
	Train,
	Predict
};

Mode mode;

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		srand(time(0));

		auto cli = (
			(( command("train").set(mode, Train),
				value("network", network_file),
				value("dataset", dataset_file),
				opt_value("weights", weights_file),
				option("-o", "--output") & value("output weights", output_file),
				option("-b", "--batch") & value("batch size", batch_size),
				option("-r", "--rate") & value("learning rate", learning_rate),
				option("-d", "--decay") & value("decay", decay),
				option("-m", "--momentum") & value("momentum", momentum)
			)
			| (
				command("predict").set(mode, Predict),
				value("network", network_file),
				value("weights", weights_file),
				value("input image", image_file),
				option("-o", "--output") & value("output image", output_file),
				option("--thresh") & value("thresh", thresh)
			))
#ifdef GPU
			, (option("--gpu") & values("gpu", gpu)) | option("--no-gpu").set(gpu, -1)
#endif
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

#ifdef GPU
		if (gpu >= 0) {
			cudaError_t status = cudaSetDevice(gpu);
			if (status != cudaSuccess) {
				throw std::runtime_error(cudaGetErrorString(status));
			}
			use_gpu = true;
		}
#endif

		switch (mode) {
			case Train:
				if (!output_file.size()) {
					output_file = "output.weights";
				}
				train();
				break;
			case Predict:
				if (!output_file.size()) {
					output_file = "output.png";
				}
				predict();
				break;
		}
	}
	catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
	return 0;
}
