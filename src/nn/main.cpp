
#include <chrono>
#include <iostream>
#include <clipp.h>
#include "OCR.hpp"

std::string image_file;
std::string query;
std::string output_file = "output.png";

void predict()
{
	stb::Image image(image_file);
	nn::OCR& ocr = nn::OCR::instance();

	auto start = std::chrono::high_resolution_clock::now();
	auto result = ocr.run(image);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	if (query.size() == 0) {
		for (auto& textline: result.textlines) {
			for (auto& word: textline.words) {
				image.draw(word.rect.left, word.rect.top, word.rect.right, word.rect.bottom, 200, 20, 50);
			}
			for (auto& char_: textline.chars) {
				std::cout << char_.alternatives[0];
			}
			std::cout << std::endl;
		}
	} else {
		auto rects = result.search(query);
		for (auto& rect: rects) {
			image.draw(rect.left, rect.top, rect.right, rect.bottom, 200, 20, 50);
		}

		std::cout << "Found: " << rects.size() << std::endl;
	}

	image.write_png(output_file);
}

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		auto cli = (
			value("input image", image_file),
			option("-q", "--query") & value("the text to search for", query),
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
