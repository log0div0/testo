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
#include "DataLoader.hpp"

void train(const std::string& cfgfile, const std::string& weightfile, const std::vector<int>& gpus)
{
	using namespace darknet;

	const std::string backup_directory = "backup/";

	float avg_loss = -1;

	Trainer trainer(cfgfile, gpus);
	if (weightfile.size()) {
		trainer.load_weights(weightfile);
	}

	DataLoader data_loader("dataset/image_list.txt", trainer);

	while (trainer.current_batch() < trainer.max_batches())
	{
		float loss = trainer.train(data_loader.load_data());

		if (avg_loss < 0) {
			avg_loss = loss;
		} else {
			avg_loss = avg_loss*.9 + loss*.1;
		}

		int i = trainer.current_batch();
		printf("%d: %f, %f avg\n", i, loss, avg_loss);

		if (i % 100 == 0) {
			trainer.save_weights(backup_directory + "/net.weights");
		}
		if (i % 10000==0 || (i < 1000 && i % 100 == 0)) {
			trainer.save_weights(backup_directory + "/net_" + std::to_string(i) + ".weights");
		}
	}
	trainer.save_weights(backup_directory + "/net_final.weights");
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

void test(const std::string& cfgfile, const std::string& weightfile, const std::string& infile, float thresh, const std::string& outfile)
{
	using namespace darknet;

	Network network(cfgfile);
	network.load_weights(weightfile);
	network.set_batch(1);

	Image image = Image(infile);

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

	image.save(outfile);
}

enum Mode {
	Help,
	Train,
	Test
};

Mode mode;
std::string cfg;
std::string weights;
std::string input;
std::string output = "prediction";
bool nogpu = false;
float thresh = 0.5f;
std::vector<int> gpus;

int main(int argc, char **argv)
{
	using namespace clipp;

	srand(time(0));

	auto cli = (
		command("help").set(mode, Help)
		| ( command("train").set(mode, Train),
			value("cfg", cfg),
			option("weights", weights)
#ifdef GPU
			,
			option("--gpus") & values("gpus", gpus)
#endif
		)
		| (
			command("test").set(mode, Test),
			value("cfg", cfg),
			value("weights", weights),
			value("input", input),
			option("-o", "--output") & value("output", output),
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
		case Help:
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			break;
		case Train:
			if (!gpus.size()) {
				gpus = {0};
			}
			train(cfg, weights, gpus);
			break;
		case Test:
			test(cfg, weights, input, thresh, output);
			break;
	}

	return 0;
}
