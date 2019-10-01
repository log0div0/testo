
#include <chrono>
#include <iostream>
#include <clipp.h>
#include <signal.h>
#include "YOLO.hpp"

std::string network_file;
std::string dataset_file;
std::string weights_file;
std::string image_file;
std::string output_file;
std::string symbols_file;
std::string query;
int batch_size = 32;
float learning_rate = 0.0001;
float momentum = 0.9;
float decay = 0.0005;
#ifdef GPU
int gpu = 0;
#endif

bool stop_training = false;

void sig_handler(int signum)
{
	stop_training = true;
}

void train()
{
	signal(SIGINT, sig_handler);

	yolo::Dataset dataset(dataset_file);

	darknet::Network network(network_file, batch_size, dataset.image_width, dataset.image_height, dataset.image_channels);
	if (weights_file.size()) {
		network.load_weights(weights_file);
	}

	float avg_loss = -1;

	for (size_t i = 0; !stop_training; ++i)
	{
		float loss = yolo::train(network, dataset, learning_rate, momentum, decay);

		if (avg_loss < 0) {
			avg_loss = loss;
		} else {
			avg_loss = avg_loss*.9 + loss*.1;
		}

		std::cout << i << ": loss = " << loss << ", avg_loss = " << avg_loss << std::endl;
	}

	network.save_weights(output_file);
}

void predict()
{
	stb::Image image(image_file);

	darknet::Network network(network_file, 1, image.width, image.height, image.channels);
	network.load_weights(weights_file);

	auto symbols = yolo::load_symbols(symbols_file);

	for (size_t i = 0; i < 10; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		yolo::predict(network, image, query, {}, {}, symbols);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;
	}

	image.write_png(output_file);
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
				value("query", query),
				value("symbols", symbols_file),
				option("-o", "--output") & value("output image", output_file)
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
