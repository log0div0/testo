
#include "TextColorPicker.hpp"
#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>
#include <stb_image_resize.h>
#include <cmath>

#define IN_H 32
#define IN_W 16

std::vector<std::string> colors = {
	"white",
	"gray",
	"black",
	"red",
	"orange",
	"yellow",
	"green",
	"cyan",
	"blue",
	"purple"
};

namespace nn {

TextColorPicker& TextColorPicker::instance() {
	static TextColorPicker instance;
	return instance;
}

TextColorPicker::TextColorPicker() {
	session = LoadModel("TextColorPicker");
}

TextColorPicker::~TextColorPicker() {

}

void TextColorPicker::run(const stb::Image* image, Char& char_) {
	run_nn(image, char_);
	return run_postprocessing(char_);
}

void TextColorPicker::run_nn(const stb::Image* image, const Char& char_) {

	if (!in_c || !out_c) {
		in_c = 3;
		out_c = colors.size() + colors.size();

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, IN_H, IN_W};
		std::array<int64_t, 2> out_shape = {1, out_c};

		in.resize(in_c * IN_H * IN_W);
		out.resize(out_c);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	int char_h = char_.rect.height();
	int char_w = char_.rect.width();
	char_img.resize(char_h * char_w * in_c);
	for (int y = 0; y < char_h; ++y) {
		for (int x = 0; x < char_w; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = (char_.rect.top + y) * image->width * image->channels + (char_.rect.left + x) * image->channels + c;
				int dst_index = y * char_w * in_c + x * in_c + c;
				char_img[dst_index] = image->data[src_index];
			}
		}
	}

	char_img_resized.resize(IN_H * IN_W * in_c);
	if (!stbir_resize_uint8(
		char_img.data(), char_w, char_h, 0,
		char_img_resized.data(), IN_W, IN_H, 0,
		in_c)
	) {
		throw std::runtime_error("stbir_resize_uint8 failed");
	}

	// std::string path = "tmp/" + std::to_string(b) + ".png";
	// if (!stbi_write_png(path.c_str(), IN_W, IN_H, in_c, char_img_resized.data(), IN_W*in_c)) {
	// 	throw std::runtime_error("Cannot save image " + path + " because " + stbi_failure_reason());
	// }

	for (int y = 0; y < IN_H; ++y) {
		for (int x = 0; x < IN_W; ++x) {
			for (int c = 0; c < in_c; ++c) {
				int src_index = y * IN_W * in_c + x * in_c + c;
				int dst_index = c * IN_H * IN_W + y * IN_W + x;
				in[dst_index] = float(char_img_resized[src_index]) / 255.0;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

void TextColorPicker::run_postprocessing(Char& char_) {
	{
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (size_t i = 0; i < colors.size(); ++i) {
			if (max_value < out[i]) {
				max_value = out[i];
				max_pos = i;
			}
		}
		char_.foreground = colors.at(max_pos);
	}
	{
		int max_pos = -1;
		float max_value = std::numeric_limits<float>::lowest();
		for (size_t i = 0; i < colors.size(); ++i) {
			if (max_value < out[colors.size()+i]) {
				max_value = out[colors.size()+i];
				max_pos = i;
			}
		}
		char_.background = colors.at(max_pos);
	}
}

}
