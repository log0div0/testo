
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

bool is_n_times_div_by_2(int value, size_t n = 4) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

int nearest_n_times_div_by_2(int value, size_t n = 4) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

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
		in_pad_h = nearest_n_times_div_by_2(in_h);
		in_pad_w = nearest_n_times_div_by_2(in_w);

		out_c = 2;
		out_h = in_h;
		out_w = in_w;
		out_pad_h = nearest_n_times_div_by_2(out_h);
		out_pad_w = nearest_n_times_div_by_2(out_w);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, in_pad_h, in_pad_w};
		std::array<int64_t, 4> out_shape = {1, out_pad_h, out_pad_w, out_c};

		in.resize(in_c * in_pad_h * in_pad_w);
		out.resize(out_c * out_pad_h * out_pad_w);

		std::fill(in.begin(), in.end(), 0);
		std::fill(out.begin(), out.end(), 0);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
    out_tensor = std::make_unique<Ort::Value>(
    	Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));

		labelingWu = LabelingWu(out_w, out_h);
	}

	float mean[3] = {0.485, 0.456, 0.406};
	float variance[3] = {0.229, 0.224, 0.225};
	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = c * in_pad_h * in_pad_w + y * in_pad_w + x;
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
			labelingWu.I[y*out_w + x] = out[y*out_pad_w*out_c + x*out_c] > .75;
		}
	}
	std::vector<Rect> rects = labelingWu.run();
	for (size_t l = 0; l < rects.size(); ++l) {
		Rect& rect = rects[l];
		Rect new_rect;

		{
			int32_t x = rect.left;
			float prev_max = 1;
			while (x > 0) {
				--x;
				float max = 0;
				for (int32_t y = rect.top; y <= rect.bottom; ++y) {
					if (max < out[y*out_pad_w*out_c + x*out_c]) {
						max = out[y*out_pad_w*out_c + x*out_c];
					}
				}
				if ((max < 0.25) || (max > prev_max)) {
					++x;
					break;
				}
				prev_max = max;
			}
			new_rect.left = x;
		}
		{
			int32_t x = rect.right;
			float prev_max = 1;
			while (x < (out_w - 1)) {
				++x;
				float max = 0;
				for (int32_t y = rect.top; y <= rect.bottom; ++y) {
					if (max < out[y*out_pad_w*out_c + x*out_c]) {
						max = out[y*out_pad_w*out_c + x*out_c];
					}
				}
				if ((max < 0.25) || (max > prev_max)) {
					--x;
					break;
				}
				prev_max = max;
			}
			new_rect.right = x;
		}
		{
			int32_t y = rect.top;
			float prev_max = 1;
			while (y > 0) {
				--y;
				float max = 0;
				for (int32_t x = rect.left; x <= rect.right; ++x) {
					if (max < out[y*out_pad_w*out_c + x*out_c]) {
						max = out[y*out_pad_w*out_c + x*out_c];
					}
				}
				if ((max < 0.5) || (max > prev_max)) {
					++y;
					break;
				}
				prev_max = max;
			}
			new_rect.top = y;
		}
		{
			int32_t y = rect.bottom;
			float prev_max = 1;
			while (y < (out_h - 1)) {
				++y;
				float max = 0;
				for (int32_t x = rect.left; x <= rect.right; ++x) {
					if (max < out[y*out_pad_w*out_c + x*out_c]) {
						max = out[y*out_pad_w*out_c + x*out_c];
					}
				}
				if ((max < 0.5) || (max > prev_max)) {
					--y;
					break;
				}
				prev_max = max;
			}
			new_rect.bottom = y;
		}

		rect = new_rect;
	}
	return rects;
}

}
