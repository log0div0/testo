
#include "TextDetector.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <onnxruntime_cxx_api.h>
#pragma GCC diagnostic pop
#include <iostream>
#include <utf8.hpp>

extern unsigned char TextDetector_onnx[];
extern unsigned int TextDetector_onnx_len;

namespace nn {

TextDetector::TextDetector() {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "text_detector");
	session = std::make_unique<Ort::Session>(*env, TextDetector_onnx, TextDetector_onnx_len, Ort::SessionOptions{nullptr});
}

TextDetector::~TextDetector() {

}

std::vector<Rect> TextDetector::detect(const stb::Image& image)
{
	if (!image.data) {
		return {};
	}

	run_nn(image);
	return find_rects();
}

void TextDetector::run_nn(const stb::Image& image) {
	if ((in_w != image.width) ||
		(in_h != image.height))
	{
		in_c = 3;
		in_h = image.height;
		in_w = image.width;

		out_c = 2;
		out_h = in_h;
		out_w = in_w;

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, in_h, in_w};
		std::array<int64_t, 4> out_shape = {1, out_h, out_w, out_c};

		in.resize(in_c * in_h * in_w);
		out.resize(out_c * out_h * out_w);
		labelingWu = LabelingWu(out_w, out_h);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
    out_tensor = std::make_unique<Ort::Value>(
    	Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));
	}

	float mean[3] = {0.485, 0.456, 0.406};
	float variance[3] = {0.229, 0.224, 0.225};
	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = c * image.height * image.width + y * image.width + x;
				in[dst_index] = float(image.data[src_index]) / 255.0f;
				in[dst_index] -= mean[c];
				in[dst_index] /= variance[c];
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

std::vector<Rect> TextDetector::find_rects() {
	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			labelingWu.I[y * out_w + x] = out[y * out_w * out_c + x * out_c + 0] > 0.75;
		}
	}
	return labelingWu.run();
}

}
