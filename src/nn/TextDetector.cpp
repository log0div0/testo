
#include "TextDetector.hpp"
#include <iostream>
#include <algorithm>

extern unsigned char TextDetector_onnx[];
extern unsigned int TextDetector_onnx_len;

namespace nn {

static inline bool is_n_times_div_by_2(int value, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

static inline int nearest_n_times_div_by_2(int value, size_t n) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

TextDetector::TextDetector() {
	session = LoadModel(TextDetector_onnx, TextDetector_onnx_len);
}

TextDetector::~TextDetector() {

}

std::vector<Word> TextDetector::detect(const stb::Image& image)
{
	if (!image.data) {
		return {};
	}

	run_nn(image);
	return find_words();
}

void TextDetector::run_nn(const stb::Image& image) {
	if ((in_w != image.width) ||
		(in_h != image.height))
	{
		in_c = 3;
		in_h = image.height;
		in_w = image.width;
		in_pad_h = nearest_n_times_div_by_2(in_h, 4);
		in_pad_w = nearest_n_times_div_by_2(in_w, 4);

		out_c = 2;
		out_h = in_h;
		out_w = in_w;
		out_pad_h = nearest_n_times_div_by_2(out_h, 4);
		out_pad_w = nearest_n_times_div_by_2(out_w, 4);

		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

		std::array<int64_t, 4> in_shape = {1, in_c, in_pad_h, in_pad_w};
		std::array<int64_t, 4> out_shape = {1, out_c, out_pad_h, out_pad_w};

		in.resize(in_c * in_pad_h * in_pad_w);
		out.resize(out_c * out_pad_h * out_pad_w);

		std::fill(in.begin(), in.end(), 0);
		std::fill(out.begin(), out.end(), 0);

		in_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, in.data(), in.size(), in_shape.data(), in_shape.size()));
		out_tensor = std::make_unique<Ort::Value>(
			Ort::Value::CreateTensor<float>(memory_info, out.data(), out.size(), out_shape.data(), out_shape.size()));

		labeling_wu[0] = LabelingWu(out_w, out_h);
		labeling_wu[1] = LabelingWu(out_w, out_h);
	}

	for (int y = 0; y < image.height; ++y) {
		for (int x = 0; x < image.width; ++x) {
			for (int c = 0; c < 3; ++c) {
				int src_index = y * image.width * image.channels + x * image.channels + c;
				int dst_index = c * in_pad_h * in_pad_w + y * in_pad_w + x;
				in[dst_index] = float(image.data[src_index]) / 255.0f;
			}
		}
	}

	const char* in_names[] = {"input"};
	const char* out_names[] = {"output"};

	session->Run(Ort::RunOptions{nullptr}, in_names, &*in_tensor, 1, out_names, &*out_tensor, 1);
}

std::vector<Word> TextDetector::find_words() {
	std::vector<Rect> up_rects = find_rects(0);
	std::vector<Rect> down_rects = find_rects(1);
	std::vector<Word> words;
	for (const Rect& up_rect: up_rects) {
		int x = up_rect.center_x();
		int y_begin = up_rect.bottom;
		int y_end = up_rect.bottom + up_rect.height();
		if (y_end > out_h) {
			y_end = out_h;
		}
		for (int y = y_begin; y < y_end; ++y) {
			uint16_t l = labeling_wu[1].L[y*out_w + x];
			if (l) {
				const Rect& down_rect = down_rects.at(l-1);
				Rect word_rect = up_rect | down_rect;
				bool found = false;
				for (auto& word: words) {
					if (word.rect.iou(word_rect) > 0.25) {
						word.rect |= word_rect;
						found = true;
						break;
					}
				}
				if (!found) {
					Word word;
					word.rect = word_rect;
					words.push_back(word);
				}
				break;
			}
		}
	}

	return words;
}

std::vector<Rect> TextDetector::find_rects(int c) {
	for (int y = 0; y < out_h; ++y) {
		for (int x = 0; x < out_w; ++x) {
			labeling_wu[c].I[y*out_w + x] = out[c*out_pad_h*out_pad_w + y*out_pad_w + x] >= .75;
		}
	}
	std::vector<Rect> rects = labeling_wu[c].run();
	for (size_t i = 0; i < rects.size(); ++i) {
		rects[i] = adjust_rect(c, rects[i]);
	}
	return rects;
}

Rect TextDetector::adjust_rect(int c, const Rect& rect) {
	Rect new_rect;

	{
		int32_t x = rect.left;
		float min_mean = 1;
		while (x > 0) {
			--x;
			float mean = 0;
			for (int32_t y = rect.top; y <= rect.bottom; ++y) {
				mean += out[c*out_pad_h*out_pad_w + y*out_pad_w + x];
			}
			mean /= rect.height();
			if ((mean < 0.25) || (mean > min_mean)) {
				++x;
				break;
			}
			if (min_mean > mean) {
				min_mean = mean;
			}
		}
		new_rect.left = x;
	}
	{
		int32_t x = rect.right;
		float min_mean = 1;
		while (x < (out_w - 1)) {
			++x;
			float mean = 0;
			for (int32_t y = rect.top; y <= rect.bottom; ++y) {
				mean += out[c*out_pad_h*out_pad_w + y*out_pad_w + x];
			}
			mean /= rect.height();
			if ((mean < 0.25) || (mean > min_mean)) {
				--x;
				break;
			}
			if (min_mean > mean) {
				min_mean = mean;
			}
		}
		new_rect.right = x;
	}
	{
		int32_t y = rect.top;
		float min_mean = 1;
		while (y > 0) {
			--y;
			float mean = 0;
			for (int32_t x = rect.left; x <= rect.right; ++x) {
				mean += out[c*out_pad_h*out_pad_w + y*out_pad_w + x];
			}
			mean /= rect.width();
			if ((mean < 0.25) || (mean > min_mean)) {
				++y;
				break;
			}
			if (min_mean > mean) {
				min_mean = mean;
			}
		}
		new_rect.top = y;
	}
	{
		int32_t y = rect.bottom;
		float min_mean = 1;
		while (y < (out_h - 1)) {
			++y;
			float mean = 0;
			for (int32_t x = rect.left; x <= rect.right; ++x) {
				mean += out[c*out_pad_h*out_pad_w + y*out_pad_w + x];
			}
			mean /= rect.width();
			if ((mean < 0.25) || (mean > min_mean)) {
				--y;
				break;
			}
			if (min_mean > mean) {
				min_mean = mean;
			}
		}
		new_rect.bottom = y;
	}

	return new_rect;
}

}
