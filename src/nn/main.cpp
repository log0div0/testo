
#include <chrono>
#include <iostream>
#include <clipp.h>
#include "OCR.hpp"

std::string image_file;
std::string output_file = "output.png";

void predict()
{
	stb::Image image(image_file);
	nn::OCR ocr;

	auto start = std::chrono::high_resolution_clock::now();
	auto textlines = ocr.run(image);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << time.count() << " seconds" << std::endl;

	for (auto& textline: textlines) {
		for (auto& word: textline.words) {
			image.draw(word.rect.left, word.rect.top, word.rect.right, word.rect.bottom, 200, 20, 50);
			std::cout << word.text << " ";
		}
		std::cout << std::endl;
	}

	image.write_png(output_file);
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
