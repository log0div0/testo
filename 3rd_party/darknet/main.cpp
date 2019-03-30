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
#include "Network.hpp"

using namespace darknet;

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

		bbox_count = std::stoi(ini.get("bbox_count"));
		bbox_size = 5;
		label_size = bbox_count * bbox_size;

		image_dir = ini.get("image_dir") + "/";
		label_dir = ini.get("label_dir") + "/";
	}

	void charge(Network* network) {
		memset(network->truth, 0, label_size*network->batch*sizeof(float));
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

			float x, y, w, h;
			size_t bbox_class;
			size_t bbox_index = 0;
			std::string label_path = label_dir + std::to_string(item_index) + ".txt";
			std::ifstream label(label_path);
			if (!label.is_open()) {
				throw std::runtime_error("Failed to open file " + label_path);
			}
			while (label >> bbox_class >> x >> y >> w >> h) {
				if (bbox_index == bbox_count) {
					throw std::runtime_error("Label of invalid size");
				}
				network->truth[label_size*row_index + bbox_size*bbox_index + 0] = x;
				network->truth[label_size*row_index + bbox_size*bbox_index + 1] = y;
				network->truth[label_size*row_index + bbox_size*bbox_index + 2] = w;
				network->truth[label_size*row_index + bbox_size*bbox_index + 3] = h;
				network->truth[label_size*row_index + bbox_size*bbox_index + 4] = bbox_class;
				++bbox_index;
			}
		}
	}

private:
	size_t item_count;
	size_t image_size, label_size;
	size_t image_width, image_height, image_channels, bbox_size, bbox_count;
	std::string image_dir, label_dir;
};

std::string network_file;
std::string dataset_file;
std::string weights_file;
std::string image_file;
std::string output_file;
float thresh = 0.5f;
#ifdef GPU
int gpu = 0;
#endif

void train()
{
	Network network(network_file);
	if (weights_file.size()) {
		network.load_weights(weights_file);
	}
	network.train = 1;

	Dataset dataset(dataset_file);

	float avg_loss = -1;

	for (size_t i = 0; ; ++i)
	{
		dataset.charge(&network);
		network.forward();
		network.backward();
		float sum = 0;
		int count = 0;
		for(size_t i = 0; i < network.n; ++i) {
			if (network.layers[i].cost) {
				sum += network.layers[i].cost[0];
				++count;
			}
		}
		float loss = sum/count/network.batch;
		network.update();

		if (avg_loss < 0) {
			avg_loss = loss;
		} else {
			avg_loss = avg_loss*.9 + loss*.1;
		}

		std::cout << i << ": loss = " << loss << ", avg_loss = " << avg_loss << std::endl;

		if (i && (i % 100 == 0)) {
			network.save_weights(output_file);
		}
	}
}

struct Box {
	uint16_t left = 0, top = 0, right = 0, bottom = 0;

	float iou(const Box& other) const {
		return float((*this & other).area()) / (*this | other).area();
	}

	uint16_t area() const {
		return (right - left) * (bottom - top);
	}

	Box operator|(const Box& other) const {
		return {
			std::min(left, other.left),
			std::min(top, other.top),
			std::max(right, other.right),
			std::max(bottom, other.bottom)
		};
	}
	Box& operator|=(const Box& other) {
		left = std::min(left, other.left);
		top = std::min(top, other.top);
		right = std::max(right, other.right);
		bottom = std::max(bottom, other.bottom);
		return *this;
	}
	Box operator&(const Box& other) const {
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
	Box& operator&=(const Box& other);

	uint16_t width() const {
		return right - left;
	}

	uint16_t height() const {
		return bottom - top;
	}
};

struct BoxSet: std::list<Box> {
	void add(const Box& box)
	{
		Box box_ext = box;
		if (box_ext.left >= 16) {
			box_ext.left -= 16;
		} else {
			box_ext.left = 0;
		}
		for (auto it = begin(); it != end(); ++it) {
			Box intersectoin = *it & box_ext;
			if (intersectoin.area()) {
				if ((intersectoin.height() * 2 > box.height()) || (intersectoin.height() * 2 > it->height())) {
					Box union_ = *it | box;
					erase(it);
					add(union_);
					return;

				}
			}
		}

		push_back(box);
	}
};

void predict()
{
	Network network(network_file);
	network.load_weights(weights_file);
	network.train = 0;

	Image image = Image(image_file);

	auto start = std::chrono::high_resolution_clock::now();

	if ((image.w * image.h * image.c) != network.inputs)
	{
		throw std::runtime_error("Image size is not equal to network size");
	}

	memcpy(network.input, image.data, network.inputs * sizeof(float));
	network.truth = 0;
	network.delta = 0;
	network.forward();
	float* predictions = network.back().output;

	// const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";

	const auto& l = network.back();

	size_t dimension_size = l.w * l.h;

	BoxSet boxes;
	for (int y = 0; y < l.h; ++y) {
		for (int x = 0; x < l.w; ++x) {
			int i = y * l.w + x;
			float objectness = predictions[dimension_size * 4 + i];
			if (objectness < thresh) {
				continue;
			}

			box b;
			b.x = (x + predictions[dimension_size * 0 + i]) / l.w;
			b.y = (y + predictions[dimension_size * 1 + i]) / l.h;
			b.w = exp(predictions[dimension_size * 2 + i]) * l.anchor_w / image.width();
			b.h = exp(predictions[dimension_size * 3 + i]) * l.anchor_h / image.height();


			Box box;
			box.left = (b.x-b.w/2)*image.width();
			box.right = (b.x+b.w/2)*image.width();
			box.top = (b.y-b.h/2)*image.height();
			box.bottom = (b.y+b.h/2)*image.height();

			boxes.add(box);
			image.draw(box.left, box.top, box.right, box.bottom, 0.9f, 0.2f, 0.3f);
		}
	}

	for (auto& box: boxes) {
		if (box.height() >= 8 && box.height() <= 24) {
			if (box.width() >= 8) {
				// image.draw(box.left, box.top, box.right, box.bottom, 0.9f, 0.2f, 0.3f);
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
				option("-o", "--output") & value("output weights", output_file)
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
					output_file = "output";
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
