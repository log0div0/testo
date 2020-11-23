
#include <chrono>
#include <iostream>
#include <clipp.h>
#include "OCR.hpp"
#include "OnnxRuntime.hpp"

std::string image_file;
std::string query;
std::string output_file = "output.png";

void draw_rect(stb::Image<stb::RGB>& img, nn::Rect bbox, stb::RGB color) {
	for (int y = bbox.top; y <= bbox.bottom; ++y) {
		img.at(bbox.left, y) = color;
		img.at(bbox.right, y) = color;
	}
	for (int x = bbox.left; x < bbox.right; ++x) {
		img.at(x, bbox.top) = color;
		img.at(x, bbox.bottom) = color;
	}
}

std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

void predict()
{
	stb::Image<stb::RGB> image(image_file);

	auto start = std::chrono::high_resolution_clock::now();
	nn::TextTensor tensor = nn::find_text(&image);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	if (query.size() == 0) {
		for (auto& textline: tensor.objects) {
			for (auto& char_: textline.chars) {
				draw_rect(image, char_.rect, {200, 20, 50});
				std::cout << conv.to_bytes(char_.codepoints[0]);
			}
			std::cout << std::endl;
		}
	} else {
		tensor = tensor.match(query);
		for (auto& textline: tensor.objects) {
			draw_rect(image, textline.rect, {200, 20, 50});
		}

		std::cout << "Found: " << tensor.objects.size() << std::endl;
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
