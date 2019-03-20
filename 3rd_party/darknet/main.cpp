#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <clipp.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <list>
#include "Network.hpp"
#include "Trainer.hpp"

using namespace darknet;

std::string network_file;
std::string dataset_file;
std::string weights_file;
std::string input_file;
std::string output_file;
float thresh = 0.5f;
#ifdef GPU
std::vector<int> gpus;
bool nogpu = false;
#endif

void train()
{
	Trainer trainer(network_file, dataset_file
#ifdef GPU
		, gpus
#endif
	);
	if (input_file.size()) {
		trainer.load_weights(input_file);
	}

	float avg_loss = -1;

	while (true)
	{
		float loss = trainer.train();

		if (avg_loss < 0) {
			avg_loss = loss;
		} else {
			avg_loss = avg_loss*.9 + loss*.1;
		}

		size_t i = trainer.current_batch();

		std::cout << i << ": loss = " << loss << ", avg_loss = " << avg_loss << std::endl;

		if (i) {
			if (i % 100 == 0) {
				trainer.save_weights(output_file);
			}
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

void test()
{
	Network network(network_file);
	network.load_weights(weights_file);
	network.set_batch(1);

	Image image = Image(input_file);

	auto start = std::chrono::high_resolution_clock::now();

	float* predictions = network.predict(image);

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
			b.w = exp(predictions[dimension_size * 2 + i]) * l.biases[0] / image.width();
			b.h = exp(predictions[dimension_size * 3 + i]) * l.biases[1] / image.height();


			Box box;
			box.left = (b.x-b.w/2)*image.width();
			box.right = (b.x+b.w/2)*image.width();
			box.top = (b.y-b.h/2)*image.height();
			box.bottom = (b.y+b.h/2)*image.height();

			boxes.add(box);
		}
	}

	for (auto& box: boxes) {
		if (box.height() >= 8 && box.height() <= 24) {
			if (box.width() >= 8) {
				image.draw(box.left, box.top, box.right, box.bottom, 0.9f, 0.2f, 0.3f);
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
	Test
};

Mode mode;

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		srand(time(0));

		auto cli = (
			( command("train").set(mode, Train),
				value("network", network_file),
				value("dataset", dataset_file),
				option("-i", "--input") & value("input weights", input_file),
				option("-o", "--output") & value("output weights", output_file)
	#ifdef GPU
				,
				option("--gpus") & values("gpus", gpus)
	#endif
			)
			| (
				command("test").set(mode, Test),
				value("network", network_file),
				value("weights", weights_file),
				value("input image", input_file),
				option("-o", "--output") & value("output image", output_file),
	#ifdef GPU
				option("--nogpu").set(nogpu),
	#endif
				option("--thresh") & value("thresh", thresh)
			)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		switch (mode) {
			case Train:
	#ifdef GPU
				if (!gpus.size()) {
					gpus = {0};
				}
	#endif
				if (!output_file.size()) {
					output_file = "output.weights";
				}
				train();
				break;
			case Test:
				if (!output_file.size()) {
					output_file = "output";
				}
				test();
				break;
		}
	}
	catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
	return 0;
}
