
#include <chrono>
#include <iostream>
#include <clipp.h>
#include "OCR.hpp"
#include "OnnxRuntime.hpp"

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

struct TextArgs {
	std::string img_file;
	std::string query;
};

void text_mode(const TextArgs& args)
{
	stb::Image<stb::RGB> image(args.img_file);

	auto start = std::chrono::high_resolution_clock::now();
	nn::TextTensor tensor = nn::find_text(&image);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	if (args.query.size() == 0) {
		for (auto& textline: tensor.objects) {
			for (auto& char_: textline.chars) {
				draw_rect(image, char_.rect, {200, 20, 50});
				std::cout << conv.to_bytes(char_.codepoints[0]);
			}
			std::cout << std::endl;
		}
	} else {
		tensor = tensor.match(args.query);
		for (auto& textline: tensor.objects) {
			draw_rect(image, textline.rect, {200, 20, 50});
		}

		std::cout << "Found: " << tensor.size() << std::endl;
	}

	image.write_png("output.png");
}

struct ImgArgs {
	std::string search_img_file;
	std::string ref_img_file;
};

void img_mode(const ImgArgs& args)
{
	stb::Image<stb::RGB> image(args.search_img_file);

	auto start = std::chrono::high_resolution_clock::now();
	nn::ImgTensor tensor = nn::find_img(&image, args.ref_img_file);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "Time: " << time.count() << " seconds" << std::endl;

	std::cout << "Found: " << tensor.size() << std::endl;

	for (auto& img: tensor.objects) {
		draw_rect(image, img.rect, {200, 20, 50});
	}

	image.write_png("output.png");
}

enum class mode {
	text,
	img,
};

int main(int argc, char **argv)
{
	try {
		using namespace clipp;

		mode selected_mode;

		TextArgs text_args;
		auto text_spec = (
			command("text").set(selected_mode, mode::text),
			value("input image", text_args.img_file),
			option("--query") & value("the text to search for", text_args.query)
		);

		ImgArgs img_args;
		auto img_spec = (
			command("img").set(selected_mode, mode::img),
			value("search image", img_args.search_img_file),
			value("ref image", img_args.ref_img_file)
		);

		auto cli = (text_spec | img_spec);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		nn::onnx::Runtime onnx_runtime;
		switch (selected_mode) {
			case mode::text:
				text_mode(text_args);
				break;
			case mode::img:
				img_mode(img_args);
				break;
			default:
				throw std::runtime_error("Invalid mode");
		}
	}
	catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
	return 0;
}
