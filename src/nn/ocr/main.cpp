
#include <chrono>
#include <iostream>
#include <clipp.h>
#include "TextDetector.hpp"
#include "TextRecognizer.hpp"

std::string image_file;
std::string output_file = "output.png";

void predict()
{
	stb::Image image(image_file);
	nn::TextDetector detector;
	nn::TextRecognizer recognizer;

	auto start = std::chrono::high_resolution_clock::now();
	auto rects = detector.detect(image);
	auto texts = recognizer.recognize(image, rects);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << time.count() << " seconds" << std::endl;

	for (auto& rect: rects) {
		image.draw(rect.left, rect.top, rect.right, rect.bottom, 200, 20, 50);
	}

	image.write_png(output_file);

	for (auto& text: texts) {
		std::cout << text << std::endl;
	}
}

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		auto cli = (
			value("input image", image_file),
			option("-o", "--output") & value("output image", output_file)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		nn::OnnxRuntime onnx_runtime;
		predict();
	}
	catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
	return 0;
}
